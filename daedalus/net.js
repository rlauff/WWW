// The network, in the browser.
//
// One file, two ways to run it. `GpuNet` puts the trunk on WebGPU, which is the
// only way this is fast enough to play with: twenty residual blocks of 256
// channels is about four hundred million multiply-adds a position, and the same
// arithmetic in JavaScript takes the better part of a second. `CpuNet` does it
// anyway, slowly, for two reasons -- it is the fallback where WebGPU is missing
// or refused, and it is the reference the shaders were written against.
//
// WHY THE REFERENCE MATTERS. The weights come off disk as a flat run of floats
// whose meaning is entirely positional; a layout read one convolution out of
// step still produces numbers, still runs, and plays like nothing at all. So
// the CPU path exists to be compared against the Rust engine on a known
// position, and the GPU path to be compared against the CPU one. See
// web/nettest.html.
//
// The activations, the leak on the value tower's single channel, and the way
// the value is derived from the margin and the spread are all copied from
// src/net.rs, which is the definition. Anywhere the two disagree, that one is
// right.

(function () {
  "use strict";

  const CELLS = 81;
  const PLANES = 12;
  const VALUE_LEAK = 0.01;          // src/net.rs VALUE_LEAK
  const MARGIN_SCALE = 20.0;        // train/serve.py and selfplay::sample
  const LOG_SIGMA_MIN = Math.log(0.01);
  const LOG_SIGMA_MAX = Math.log(2.0);

  // ---- the weight file ----------------------------------------------------

  function parseDaed(buffer) {
    const head = new DataView(buffer);
    const magic = String.fromCharCode(head.getUint8(0), head.getUint8(1),
                                      head.getUint8(2), head.getUint8(3));
    if (magic !== "DAED") throw new Error("not a weight file: the magic is wrong");
    const version = head.getUint32(4, true);
    if (version !== 2) {
      throw new Error("weight file version " + version + ", this build reads 2");
    }
    const channels = head.getUint32(8, true);
    const blocks = head.getUint32(12, true);
    const planes = head.getUint32(16, true);
    const actions = head.getUint32(20, true);
    const hidden = head.getUint32(24, true);
    if (planes !== PLANES) {
      throw new Error("the file wants " + planes + " planes, this build has " + PLANES);
    }

    const floats = new Float32Array(buffer, 28);
    let at = 0;
    const take = n => {
      if (at + n > floats.length) {
        throw new Error("the weight file ends after " + floats.length
                        + " floats, " + (at + n) + " were wanted");
      }
      const slice = floats.subarray(at, at + n);
      at += n;
      return slice;
    };
    // Exactly the order src/net.rs reads them in. Nothing here may be reordered
    // without reordering that.
    const conv = (out, inp, k) => ({ weight: take(out * inp * k * k), bias: take(out),
                                     out, inp, k });
    const linear = (out, inp) => ({ weight: take(out * inp), bias: take(out), out, inp });

    const net = { channels, blocks, hidden, actions, version };
    net.stem = conv(channels, PLANES, 3);
    net.residual = [];
    for (let i = 0; i < blocks; i++) {
      net.residual.push([conv(channels, channels, 3), conv(channels, channels, 3)]);
    }
    net.policy_conv = conv(2, channels, 1);
    net.policy_fc = linear(actions, 2 * CELLS);
    net.value_conv = conv(1, channels, 1);
    net.value_fc1 = linear(hidden, CELLS);
    net.spread = linear(1, hidden);
    net.traversal = conv(2, channels, 1);
    net.wall_map = conv(2, channels, 1);
    net.margin = linear(1, hidden);
    if (at !== floats.length) {
      throw new Error("the weight file has " + (floats.length - at)
                      + " floats left over, the shapes do not match");
    }
    return net;
  }

  // ---- what the heads mean ------------------------------------------------

  // Abramowitz and Stegun 7.1.26, the same approximation src/net.rs uses, so
  // the two agree to well under what half precision costs anyway.
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return sign * y;
  }
  const sigmaFromLogit = spread => {
    const t = 1 / (1 + Math.exp(-spread));
    return Math.exp(LOG_SIGMA_MIN + (LOG_SIGMA_MAX - LOG_SIGMA_MIN) * t);
  };
  const valueFromSpread = (mu, spread) =>
    erf(mu / (sigmaFromLogit(spread) * Math.SQRT2));

  /// Turns one position's raw head outputs into what the search wants. Shared
  /// by both backends so the derivation cannot differ between them.
  function readHeads(logits, hidden, net, out, index) {
    let logSigma = net.spread.bias[0];
    for (let i = 0; i < net.hidden; i++) logSigma += net.spread.weight[i] * hidden[i];
    let mu = net.margin.bias[0];
    for (let i = 0; i < net.hidden; i++) mu += net.margin.weight[i] * hidden[i];
    out.values[index] = valueFromSpread(mu, logSigma);
    out.margins[index] = mu * MARGIN_SCALE;
    out.sigmas[index] = sigmaFromLogit(logSigma) * MARGIN_SCALE;
  }

  // ---- the reference ------------------------------------------------------

  class CpuNet {
    constructor(net) { this.net = net; this.backend = "cpu"; }

    // out.logits is batch * actions, the rest one per position.
    forward(batch, planes, out) {
      const net = this.net, C = net.channels;
      for (let b = 0; b < batch; b++) {
        let cur = this.conv(net.stem, planes.subarray(b * PLANES * CELLS, (b + 1) * PLANES * CELLS), PLANES);
        relu(cur);
        for (const [first, second] of net.residual) {
          const skip = cur;
          let mid = this.conv(first, cur, C); relu(mid);
          cur = this.conv(second, mid, C);
          for (let i = 0; i < cur.length; i++) cur[i] += skip[i];
          relu(cur);
        }
        const head = this.conv(net.policy_conv, cur, C); relu(head);
        const logits = out.logits.subarray(b * net.actions, (b + 1) * net.actions);
        affine(net.policy_fc, head, logits);

        const plane = this.conv(net.value_conv, cur, C);
        for (let i = 0; i < plane.length; i++) if (plane[i] < 0) plane[i] *= VALUE_LEAK;
        const hidden = new Float32Array(net.hidden);
        affine(net.value_fc1, plane, hidden);
        relu(hidden);
        readHeads(logits, hidden, net, out, b);
      }
      return out;
    }

    /// The overlay's heads for one position: where each pawn is expected to
    /// walk, which walls survive, and the margin. Raw convolution outputs --
    /// the logistic is applied on the engine side so both builds squash with
    /// the same function.
    heads(planes) {
      const net = this.net, C = net.channels;
      let cur = this.conv(net.stem, planes.subarray(0, PLANES * CELLS), PLANES);
      relu(cur);
      for (const [first, second] of net.residual) {
        const skip = cur;
        let mid = this.conv(first, cur, C); relu(mid);
        cur = this.conv(second, mid, C);
        for (let i = 0; i < cur.length; i++) cur[i] += skip[i];
        relu(cur);
      }
      const traversal = this.conv(net.traversal, cur, C);
      const walls = this.conv(net.wall_map, cur, C);
      const plane = this.conv(net.value_conv, cur, C);
      for (let i = 0; i < plane.length; i++) if (plane[i] < 0) plane[i] *= VALUE_LEAK;
      const hidden = new Float32Array(net.hidden);
      affine(net.value_fc1, plane, hidden);
      relu(hidden);
      let mu = net.margin.bias[0];
      for (let i = 0; i < net.hidden; i++) mu += net.margin.weight[i] * hidden[i];
      return { traversal, walls, margin: mu * MARGIN_SCALE };
    }

    conv(layer, input, inC) {
      const out = new Float32Array(layer.out * CELLS);
      const k = layer.k, half = (k - 1) >> 1;
      for (let oc = 0; oc < layer.out; oc++) {
        const bias = layer.bias[oc];
        for (let cell = 0; cell < CELLS; cell++) {
          const y = (cell / 9) | 0, x = cell % 9;
          let sum = bias;
          for (let ic = 0; ic < inC; ic++) {
            const wbase = ((oc * inC) + ic) * k * k;
            const ibase = ic * CELLS;
            for (let ky = 0; ky < k; ky++) {
              const iy = y + ky - half;
              if (iy < 0 || iy > 8) continue;
              for (let kx = 0; kx < k; kx++) {
                const ix = x + kx - half;
                if (ix < 0 || ix > 8) continue;
                sum += layer.weight[wbase + ky * k + kx] * input[ibase + iy * 9 + ix];
              }
            }
          }
          out[oc * CELLS + cell] = sum;
        }
      }
      return out;
    }
  }

  function relu(values) {
    for (let i = 0; i < values.length; i++) if (values[i] < 0) values[i] = 0;
  }
  function affine(layer, input, out) {
    for (let o = 0; o < layer.out; o++) {
      let sum = layer.bias[o];
      const base = o * layer.inp;
      for (let i = 0; i < layer.inp; i++) sum += layer.weight[base + i] * input[i];
      out[o] = sum;
    }
  }

  // ---- WebGPU -------------------------------------------------------------
  //
  // Two shaders do the whole network. A convolution, parameterised by a uniform
  // so the same pipeline serves the 3x3 trunk and every 1x1 head, and a matrix
  // multiply for the fully connected layers. Everything is NCHW f32: batch,
  // then channel, then the 81 cells in row order, which is the layout the
  // weight file is already in and so needs no shuffling on upload.
  //
  // The last few numbers -- the spread and the margin off the 64 wide hidden
  // layer, and the error function that turns them into a value -- are read back
  // and finished in JavaScript. They are two dot products of length 64 per
  // position, far too small to be worth a dispatch, and doing them there means
  // the derivation is the one `CpuNet` was checked against rather than a second
  // copy in a shader.

  const CONV_WGSL = `
struct Dims {
  batch: u32, in_c: u32, out_c: u32, ksize: u32,
  act: u32, residual: u32, pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;
@group(0) @binding(5) var<storage, read> skip: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = dims.batch * dims.out_c * 81u;
  let index = gid.x;
  if (index >= total) { return; }

  let cell = index % 81u;
  let oc = (index / 81u) % dims.out_c;
  let b = index / (81u * dims.out_c);
  let y = i32(cell / 9u);
  let x = i32(cell % 9u);
  let half = i32(dims.ksize) / 2;

  var sum = bias[oc];
  let in_base = b * dims.in_c * 81u;
  for (var ic = 0u; ic < dims.in_c; ic = ic + 1u) {
    let w_base = (oc * dims.in_c + ic) * dims.ksize * dims.ksize;
    let i_base = in_base + ic * 81u;
    for (var ky = 0u; ky < dims.ksize; ky = ky + 1u) {
      let iy = y + i32(ky) - half;
      if (iy < 0 || iy > 8) { continue; }
      for (var kx = 0u; kx < dims.ksize; kx = kx + 1u) {
        let ix = x + i32(kx) - half;
        if (ix < 0 || ix > 8) { continue; }
        sum = sum + weight[w_base + ky * dims.ksize + kx]
                  * input[i_base + u32(iy) * 9u + u32(ix)];
      }
    }
  }
  if (dims.residual == 1u) { sum = sum + skip[index]; }
  if (dims.act == 1u) { sum = max(sum, 0.0); }
  else if (dims.act == 2u) { if (sum < 0.0) { sum = sum * ${VALUE_LEAK}; } }
  output[index] = sum;
}`;

  const LINEAR_WGSL = `
struct Dims { batch: u32, in_n: u32, out_n: u32, act: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = dims.batch * dims.out_n;
  let index = gid.x;
  if (index >= total) { return; }
  let o = index % dims.out_n;
  let b = index / dims.out_n;
  var sum = bias[o];
  let w_base = o * dims.in_n;
  let i_base = b * dims.in_n;
  for (var i = 0u; i < dims.in_n; i = i + 1u) {
    sum = sum + weight[w_base + i] * input[i_base + i];
  }
  if (dims.act == 1u) { sum = max(sum, 0.0); }
  output[index] = sum;
}`;

  class GpuNet {
    constructor(device, net, maxBatch) {
      this.device = device;
      this.net = net;
      this.maxBatch = maxBatch;
      this.backend = "webgpu";
      this.build();
    }

    static async create(net, maxBatch) {
      if (!navigator.gpu) throw new Error("this browser has no WebGPU");
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("no WebGPU adapter; the card may be blocked");
      const device = await adapter.requestDevice();
      return new GpuNet(device, net, maxBatch || 16);
    }

    storage(data) {
      const buffer = this.device.createBuffer({
        size: Math.max(4, data.byteLength),
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
      });
      new Float32Array(buffer.getMappedRange()).set(data);
      buffer.unmap();
      return buffer;
    }

    scratch(floats) {
      return this.device.createBuffer({
        size: Math.max(4, floats * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    uniform(values) {
      const buffer = this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true,
      });
      const view = new Uint32Array(buffer.getMappedRange());
      values.forEach((v, i) => { view[i] = v >>> 0; });
      buffer.unmap();
      return buffer;
    }

    build() {
      const d = this.device, net = this.net, C = net.channels, B = this.maxBatch;
      this.conv = d.createComputePipeline({
        layout: "auto",
        compute: { module: d.createShaderModule({ code: CONV_WGSL }), entryPoint: "main" },
      });
      this.linear = d.createComputePipeline({
        layout: "auto",
        compute: { module: d.createShaderModule({ code: LINEAR_WGSL }), entryPoint: "main" },
      });

      // Every weight, uploaded once -- 94 MB of f32 for a 256x20 network and
      // about 142 MB for a 256x30; the card holds it for the life of the page.
      const put = layer => ({ ...layer, w: this.storage(layer.weight), b: this.storage(layer.bias) });
      this.w = {
        stem: put(net.stem),
        residual: net.residual.map(([a, b]) => [put(a), put(b)]),
        policy_conv: put(net.policy_conv), policy_fc: put(net.policy_fc),
        value_conv: put(net.value_conv), value_fc1: put(net.value_fc1),
        traversal: put(net.traversal), wall_map: put(net.wall_map),
      };

      // Ping-pong through the trunk. Three buffers is the whole working set:
      // what the block reads, what it writes, and the skip it has to add back.
      this.bufIn = this.scratch(B * PLANES * CELLS);
      this.t0 = this.scratch(B * C * CELLS);
      this.t1 = this.scratch(B * C * CELLS);
      this.t2 = this.scratch(B * C * CELLS);
      this.bufHead = this.scratch(B * 2 * CELLS);
      this.bufPlane = this.scratch(B * CELLS);
      this.bufLogits = this.scratch(B * net.actions);
      this.bufHidden = this.scratch(B * net.hidden);
      this.bufAux0 = this.scratch(2 * CELLS);
      this.bufAux1 = this.scratch(2 * CELLS);
      this.zero = this.storage(new Float32Array(1));
      this.readAux0 = d.createBuffer({ size: 2 * CELLS * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.readAux1 = d.createBuffer({ size: 2 * CELLS * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.readLogits = d.createBuffer({
        size: B * net.actions * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.readHidden = d.createBuffer({
        size: B * net.hidden * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.dims = new Map();
    }

    dimsFor(key, values) {
      if (!this.dims.has(key)) this.dims.set(key, this.uniform(values));
      return this.dims.get(key);
    }

    runConv(pass, layer, input, output, batch, act, skip) {
      const key = [layer.inp, layer.out, layer.k, act, skip ? 1 : 0, batch].join(":");
      const dims = this.dimsFor("c" + key,
        [batch, layer.inp, layer.out, layer.k, act, skip ? 1 : 0, 0, 0]);
      const group = this.device.createBindGroup({
        layout: this.conv.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input } },
          { binding: 1, resource: { buffer: layer.w } },
          { binding: 2, resource: { buffer: layer.b } },
          { binding: 3, resource: { buffer: output } },
          { binding: 4, resource: { buffer: dims } },
          { binding: 5, resource: { buffer: skip || this.zero } },
        ],
      });
      pass.setPipeline(this.conv);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(batch * layer.out * CELLS / 64));
    }

    runLinear(pass, layer, input, output, batch, act) {
      const dims = this.dimsFor("l" + [layer.inp, layer.out, act, batch].join(":"),
                                [batch, layer.inp, layer.out, act]);
      const group = this.device.createBindGroup({
        layout: this.linear.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input } },
          { binding: 1, resource: { buffer: layer.w } },
          { binding: 2, resource: { buffer: layer.b } },
          { binding: 3, resource: { buffer: output } },
          { binding: 4, resource: { buffer: dims } },
        ],
      });
      pass.setPipeline(this.linear);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(batch * layer.out / 64));
    }

    /// The trunk, dispatched into `pass`. Returns the buffer holding it.
    trunk(pass, batch) {
      const w = this.w;
      this.runConv(pass, w.stem, this.bufIn, this.t0, batch, 1, null);
      let a = this.t0, b = this.t1, c = this.t2;
      for (const [first, second] of w.residual) {
        this.runConv(pass, first, a, b, batch, 1, null);
        this.runConv(pass, second, b, c, batch, 1, a);
        const spare = a; a = c; c = b; b = spare;
      }
      return a;
    }

    /// The overlay's heads for one position. Same trunk, two more 1x1
    /// convolutions, and the margin off the hidden layer.
    async heads(planes) {
      const d = this.device, w = this.w, net = this.net;
      d.queue.writeBuffer(this.bufIn, 0, planes, 0, PLANES * CELLS);
      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();
      const trunk = this.trunk(pass, 1);
      this.runConv(pass, w.traversal, trunk, this.bufAux0, 1, 0, null);
      this.runConv(pass, w.wall_map, trunk, this.bufAux1, 1, 0, null);
      this.runConv(pass, w.value_conv, trunk, this.bufPlane, 1, 2, null);
      this.runLinear(pass, w.value_fc1, this.bufPlane, this.bufHidden, 1, 1);
      pass.end();
      encoder.copyBufferToBuffer(this.bufAux0, 0, this.readAux0, 0, 2 * CELLS * 4);
      encoder.copyBufferToBuffer(this.bufAux1, 0, this.readAux1, 0, 2 * CELLS * 4);
      encoder.copyBufferToBuffer(this.bufHidden, 0, this.readHidden, 0, net.hidden * 4);
      d.queue.submit([encoder.finish()]);
      await Promise.all([
        this.readAux0.mapAsync(GPUMapMode.READ, 0, 2 * CELLS * 4),
        this.readAux1.mapAsync(GPUMapMode.READ, 0, 2 * CELLS * 4),
        this.readHidden.mapAsync(GPUMapMode.READ, 0, net.hidden * 4),
      ]);
      const traversal = new Float32Array(this.readAux0.getMappedRange(0, 2 * CELLS * 4).slice(0));
      const walls = new Float32Array(this.readAux1.getMappedRange(0, 2 * CELLS * 4).slice(0));
      const hidden = new Float32Array(this.readHidden.getMappedRange(0, net.hidden * 4).slice(0));
      this.readAux0.unmap(); this.readAux1.unmap(); this.readHidden.unmap();
      let mu = net.margin.bias[0];
      for (let i = 0; i < net.hidden; i++) mu += net.margin.weight[i] * hidden[i];
      return { traversal, walls, margin: mu * MARGIN_SCALE };
    }

    async forward(batch, planes, out) {
      if (batch > this.maxBatch) {
        throw new Error("this batch is " + batch + " but the card's buffers hold "
          + this.maxBatch + "; the engine's batch setting is above what the network"
          + " was uploaded for");
      }
      const d = this.device, w = this.w;
      d.queue.writeBuffer(this.bufIn, 0, planes, 0, batch * PLANES * CELLS);

      const encoder = d.createCommandEncoder();
      const pass = encoder.beginComputePass();

      // THREE trunk buffers, rotating. The obvious two are not enough: a block
      // adds its own input back at the end, so that input has to survive both
      // convolutions, and writing the second one into it would be reading and
      // writing the same buffer in one dispatch. Rotating a third means the
      // skip is always a buffer nobody is writing -- and that the whole trunk
      // is one compute pass with no copies in the middle of it, which a
      // copyBufferToBuffer would have forced (a copy is an encoder command and
      // would have ended the pass twenty times over).
      const a = this.trunk(pass, batch);

      this.runConv(pass, w.policy_conv, a, this.bufHead, batch, 1, null);
      this.runLinear(pass, w.policy_fc, this.bufHead, this.bufLogits, batch, 0);
      this.runConv(pass, w.value_conv, a, this.bufPlane, batch, 2, null);
      this.runLinear(pass, w.value_fc1, this.bufPlane, this.bufHidden, batch, 1);
      pass.end();

      encoder.copyBufferToBuffer(this.bufLogits, 0, this.readLogits, 0,
                                 batch * this.net.actions * 4);
      encoder.copyBufferToBuffer(this.bufHidden, 0, this.readHidden, 0,
                                 batch * this.net.hidden * 4);
      d.queue.submit([encoder.finish()]);

      await Promise.all([
        this.readLogits.mapAsync(GPUMapMode.READ, 0, batch * this.net.actions * 4),
        this.readHidden.mapAsync(GPUMapMode.READ, 0, batch * this.net.hidden * 4),
      ]);
      const logits = new Float32Array(
        this.readLogits.getMappedRange(0, batch * this.net.actions * 4).slice(0));
      const hidden = new Float32Array(
        this.readHidden.getMappedRange(0, batch * this.net.hidden * 4).slice(0));
      this.readLogits.unmap();
      this.readHidden.unmap();

      out.logits.set(logits.subarray(0, batch * this.net.actions));
      for (let i = 0; i < batch; i++) {
        readHeads(null, hidden.subarray(i * this.net.hidden, (i + 1) * this.net.hidden),
                  this.net, out, i);
      }
      return out;
    }
  }

  window.DAEDALUS_NET = { parseDaed, CpuNet, GpuNet, readHeads, erf, sigmaFromLogit,
                          valueFromSpread, CELLS, PLANES, MARGIN_SCALE, VALUE_LEAK };
})();
