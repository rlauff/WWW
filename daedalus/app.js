// The browser's engine: the same Rust, compiled to wasm, in place of the socket.
//
// The page is written against one seam -- a command string in, a JSON object
// out -- so this file only has to answer the same way `POST /cmd` does. When
// the wasm module is present it takes over; when it is not (running against the
// native server during development) `window.DAEDALUS_WASM` stays null and the
// page keeps talking HTTP. One index.html, both worlds.
//
// A NOTE ON THE ENGINE COUNT. The served page spawns three engine processes --
// one a side and one for analysis -- because a process is cheap. A tab is not:
// there is one wasm instance and one session in it. The page's role machinery
// is therefore pointed at the same session three times, and `broadcast` sends
// once instead of four times, which is what `DAEDALUS_WASM.single` says.

(function () {
  "use strict";

  const WASM_URL = "daedalus.wasm";

  let exports = null;
  let memory = null;
  let lastPanic = null;

  // The module's memory can be replaced when it grows, so the view is taken
  // fresh every time rather than cached. A stale view reads freed pages.
  const bytes = () => new Uint8Array(memory.buffer);
  const words = () => new DataView(memory.buffer);

  function put(text) {
    const encoded = new TextEncoder().encode(text);
    const ptr = exports.daed_alloc(encoded.length);
    bytes().set(encoded, ptr);
    return { ptr, len: encoded.length };
  }

  /// Reads a reply block: four bytes of length, then UTF-8, then hands it back.
  function take(ptr) {
    const len = words().getUint32(ptr, true);
    const text = new TextDecoder().decode(
      new Uint8Array(memory.buffer, ptr + 4, len));
    exports.daed_free(ptr, 4 + len);
    return text;
  }

  function trapped(error) {
    // A wasm trap arrives as a bare RuntimeError; the panic hook has already
    // recorded what it really was.
    const why = lastPanic ? lastPanic : (error && error.message) || String(error);
    lastPanic = null;
    return new Error(why);
  }

  function cmd(line) {
    const input = put(line);
    let out;
    try {
      out = exports.daed_cmd(input.ptr, input.len);
    } catch (error) {
      throw trapped(error);
    } finally {
      exports.daed_free(input.ptr, input.len);
    }
    const text = take(out);
    try {
      return JSON.parse(text);
    } catch (e) {
      return { ok: false, error: "the engine returned something that is not JSON: "
                                 + text.slice(0, 200) };
    }
  }

  function snapshot() {
    return JSON.parse(take(exports.daed_snapshot()));
  }

  async function start() {
    const imports = {
      env: {
        // The only thing the module asks the page for. wasm32-unknown-unknown
        // has no clock of its own; see src/wasmclock.rs.
        daedalus_now_ms: () => performance.now(),
        // A panic on this target is a bare `unreachable` trap: without this the
        // page would only ever learn that something went wrong, never what or
        // where. The message carries the file and line.
        daedalus_panic: (ptr, len) => {
          lastPanic = new TextDecoder().decode(new Uint8Array(memory.buffer, ptr, len));
          console.error("daedalus panicked: " + lastPanic);
        },
      },
    };
    let instance;
    if (WebAssembly.instantiateStreaming) {
      const answer = await WebAssembly.instantiateStreaming(fetch(WASM_URL), imports);
      instance = answer.instance;
    } else {
      const raw = await fetch(WASM_URL).then(r => r.arrayBuffer());
      const answer = await WebAssembly.instantiate(raw, imports);
      instance = answer.instance;
    }
    exports = instance.exports;
    memory = exports.memory;

    // The native build runs the background analysis on its own thread. Here the
    // page owns the only thread there is, so it lends it back a slice at a time
    // -- the same bounded pass, driven from a timer instead of a loop.
    let slicing = false;
    setInterval(() => {
      if (slicing) return;
      try {
        // 0 = nothing to do, 1 = a slice was run, 2 = the network's turn, which
        // this process cannot do because the network is not in it.
        const answer = exports.daed_analyse_slice();
        if (answer !== 2) return;
      } catch (error) {
        console.error(trapped(error).message);
        return;
      }
      // Analysis with the network is the same driven loop a move uses, run for
      // one slice at a time. The tree is kept between them, so it deepens pass
      // after pass exactly as the native analysis does.
      slicing = true;
      driveSlice().catch(error => console.error(error.message))
                  .finally(() => { slicing = false; });
    }, 60);

    // The driven search, for when a network is doing the evaluating. Each of
    // these is one step of the loop in `searchMove` below.
    const beginSearch = sims => exports.daed_search_begin(sims >>> 0) !== 0;
    const searchCount = () => exports.daed_search_count();
    function searchPlanes() {
      const ptr = exports.daed_search_planes();
      const count = words().getUint32(ptr, true);
      // Copied out rather than viewed: the next allocation can grow the
      // module's memory, and a view onto the old buffer is then detached.
      const planes = new Float32Array(
        memory.buffer.slice(ptr + 4, ptr + 4 + count * 4));
      exports.daed_free(ptr, 4 + count * 4);
      return planes;
    }
    function deliver(out) {
      const put = array => {
        if (!array || !array.length) return { ptr: 0, len: 0 };
        const ptr = exports.daed_alloc(array.length * 4);
        new Float32Array(memory.buffer, ptr, array.length).set(array);
        return { ptr, len: array.length };
      };
      const l = put(out.logits), v = put(out.values);
      const m = put(out.margins), s = put(out.sigmas);
      exports.daed_search_deliver(l.ptr, l.len, v.ptr, v.len,
                                  m.ptr, m.len, s.ptr, s.len);
      for (const b of [l, v, m, s]) if (b.ptr) exports.daed_free(b.ptr, b.len * 4);
    }
    const searchDone = () => exports.daed_search_done() !== 0;
    const searchFinish = (play, ms) =>
      JSON.parse(take(exports.daed_search_finish(play ? 1 : 0, ms >>> 0)));

    // --- the network ------------------------------------------------------
    //
    // Loaded on demand: the file is ninety odd megabytes and a visitor who only
    // wants to look at a puzzle should not pay for it. Once fetched it is kept
    // in IndexedDB, so the second visit is a disk read rather than a download.

    const WEIGHTS_URL = "weights/champ.daed";
    const DB_NAME = "daedalus", STORE = "weights";
    let net = null;                 // the evaluator, once it is up
    let netLoading = null;          // the promise, so two callers share one load
    let netNote = "";

    function idb() {
      return new Promise((resolve, reject) => {
        const open = indexedDB.open(DB_NAME, 1);
        open.onupgradeneeded = () => open.result.createObjectStore(STORE);
        open.onsuccess = () => resolve(open.result);
        open.onerror = () => reject(open.error);
      });
    }

    async function cached(key) {
      try {
        const db = await idb();
        return await new Promise((resolve, reject) => {
          const ask = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
          ask.onsuccess = () => resolve(ask.result || null);
          ask.onerror = () => reject(ask.error);
        });
      } catch (e) { return null; }     // private window, quota, no matter
    }

    async function keep(key, value) {
      try {
        const db = await idb();
        await new Promise((resolve, reject) => {
          const put = db.transaction(STORE, "readwrite").objectStore(STORE).put(value, key);
          put.onsuccess = () => resolve();
          put.onerror = () => reject(put.error);
        });
      } catch (e) { /* the next visit downloads again, which is not fatal */ }
    }

    async function weightBytes(onProgress) {
      const stored = await cached(WEIGHTS_URL);
      if (stored) { netNote = "from this browser's cache"; return stored; }
      const answer = await fetch(WEIGHTS_URL);
      if (!answer.ok) {
        throw new Error("no network at " + WEIGHTS_URL + " (HTTP " + answer.status + ")");
      }
      // Streamed so the page can say how far along it is; a silent ninety
      // megabyte wait looks like a hang.
      const total = Number(answer.headers.get("content-length") || 0);
      const reader = answer.body && answer.body.getReader ? answer.body.getReader() : null;
      let bytes;
      if (!reader) {
        bytes = await answer.arrayBuffer();
      } else {
        const chunks = [];
        let seen = 0;
        for (;;) {
          const step = await reader.read();
          if (step.done) break;
          chunks.push(step.value);
          seen += step.value.length;
          if (onProgress) onProgress(seen, total);
        }
        const joined = new Uint8Array(seen);
        let at = 0;
        for (const chunk of chunks) { joined.set(chunk, at); at += chunk.length; }
        bytes = joined.buffer;
      }
      netNote = "downloaded";
      await keep(WEIGHTS_URL, bytes);
      return bytes;
    }

    function loadNetwork(onProgress) {
      if (net) return Promise.resolve(net);
      if (netLoading) return netLoading;
      netLoading = (async () => {
        const N = window.DAEDALUS_NET;
        if (!N) throw new Error("net.js is not on the page");
        const parsed = N.parseDaed(await weightBytes(onProgress));
        try {
          net = await N.GpuNet.create(parsed, 16);
        } catch (error) {
          // No card, or the browser will not hand one over. The reference is
          // CORRECT but about six seconds a position, which at any sane time
          // budget means the search evaluates the root and nothing else. That
          // is not "a bit slower", it is a different program -- so it is said
          // plainly, here and on the page, rather than left to be inferred from
          // the moves.
          net = new N.CpuNet(parsed);
          netNote = "NO WEBGPU (" + error.message + ") — running the network on "
                  + "the cpu at seconds a move, which is far too slow to play well";
          net.slow = true;
        }
        return net;
      })();
      return netLoading;
    }

    /// One move, with the page driving the search and the card answering it.
    ///
    /// Never two at once: a driven search holds leaves on the tree that only
    /// its own `deliver` can hand back, and starting a second over the top of
    /// it corrupts the arena.
    let searching = false;
    async function searchMove(play, sims, onProgress) {
      if (searching) throw new Error("a search is already running");
      const evaluator = await loadNetwork(onProgress);
      searching = true;
      try {
        return await runSearch(evaluator, play, sims);
      } finally {
        searching = false;
      }
    }

    /// One analysis pass: the same loop, told not to play the move it finds.
    async function driveSlice() {
      if (searching || !net) return;
      searching = true;
      try {
        await runSearch(net, false, 0);
      } finally {
        searching = false;
      }
    }

    async function runSearch(evaluator, play, sims) {
      const started = performance.now();
      if (!beginSearch(sims || 0)) return { ok: false, error: "the game is over" };
      const out = {
        logits: new Float32Array(16 * evaluator.net.actions),
        values: new Float32Array(16), margins: new Float32Array(16),
        sigmas: new Float32Array(16),
      };
      for (let round = 0; round < 100000; round++) {
        const planes = searchPlanes();
        const count = searchCount();
        if (!count) break;
        await evaluator.forward(count, planes, out);
        deliver({
          logits: out.logits.subarray(0, count * evaluator.net.actions),
          values: out.values.subarray(0, count),
          margins: out.margins.subarray(0, count),
          sigmas: out.sigmas.subarray(0, count),
        });
        if (searchDone()) break;
      }
      return searchFinish(play, Math.round(performance.now() - started));
    }

    // Whether the session is set to the network. Asked of the engine rather
    // than remembered here, so a `brain` command from anywhere -- the page, a
    // preset, a reload -- is reflected without this file having to watch for it.
    function usingNet() {
      try {
        const state = cmd("state");
        return typeof state.brain === "string" && state.brain.startsWith("network on the card");
      } catch (e) { return false; }
    }

    /// The overlay's heads, computed wherever the network is and formatted by
    /// the engine, so the browser's overlay is the native one.
    async function headsNow() {
      const evaluator = await loadNetwork();
      const ptr = exports.daed_heads_planes();
      const count = words().getUint32(ptr, true);
      const planes = new Float32Array(memory.buffer.slice(ptr + 4, ptr + 4 + count * 4));
      exports.daed_free(ptr, 4 + count * 4);
      const h = await evaluator.heads(planes);
      const put = a => {
        const p = exports.daed_alloc(a.length * 4);
        new Float32Array(memory.buffer, p, a.length).set(a);
        return { ptr: p, len: a.length };
      };
      const t = put(h.traversal), w = put(h.walls);
      const out = take(exports.daed_heads_json(t.ptr, t.len, w.ptr, w.len, h.margin));
      exports.daed_free(t.ptr, t.len * 4);
      exports.daed_free(w.ptr, w.len * 4);
      return JSON.parse(out);
    }

    const netReady = () => !!net;
    const netStatus = () => netNote;

    window.DAEDALUS_WASM = { cmd, snapshot, single: true, beginSearch, searchCount,
                             searchPlanes, deliver, searchDone, searchFinish,
                             loadNetwork, searchMove, netReady, netStatus, usingNet, headsNow };
    window.dispatchEvent(new Event("daedalus-ready"));
  }

  window.DAEDALUS_BOOT = start().catch(error => {
    // Leaving the flag unset is the signal to fall back to HTTP, which is what
    // the development server wants; a page with neither says so on its face
    // rather than looking merely broken.
    console.error("the wasm engine did not start:", error);
    window.DAEDALUS_WASM = null;
    window.DAEDALUS_ERROR = String(error && error.message || error);
  });
})();
