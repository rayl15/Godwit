import Foundation

/// The dashboard, embedded as a string.
///
/// One file, no assets, no build step — the binary serves the whole interface.
/// That matters for something people are meant to clone and run: a dashboard
/// that needs `npm install` is a dashboard most people never see.
enum WebUI {
    static func page(model: String, layers: Int, experts: Int, topK: Int,
                     slots: Int, cacheGiB: Double, trunkGiB: Double) -> String {
        """
        <!doctype html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Godwit</title>
        <style>
        :root {
          --bg: #080b0f; --panel: #10151c; --line: #1e2630;
          --text: #e6edf3; --dim: #7d8b9a; --faint: #4a5764;
          --accent: #22d3ee; --accent-dim: #0e5f6b; --good: #34d399;
          /* The decay trail in the expert grid is its own colour, not a reuse
             of --accent-dim. The grid's whole claim is that ~3% of cells fire
             per token, so the trail has to sit close to the unlit cell or the
             view says the opposite of what it means. */
          --trail: #10333c;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0; background: var(--bg); color: var(--text);
          font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
          height: 100vh; display: flex; overflow: hidden;
        }
        aside {
          width: 260px; flex: 0 0 260px; background: var(--panel);
          border-right: 1px solid var(--line); padding: 16px;
          overflow-y: auto; display: flex; flex-direction: column; gap: 18px;
        }
        .brand { display: flex; align-items: baseline; gap: 8px; }
        .brand h1 { font-size: 17px; margin: 0; letter-spacing: .5px; color: var(--accent); }
        .brand span { font-size: 10px; color: var(--faint); letter-spacing: .8px; }
        h2 {
          font-size: 10px; letter-spacing: 1.2px; color: var(--faint);
          margin: 0 0 8px; text-transform: uppercase; font-weight: 500;
        }
        .row { display: flex; justify-content: space-between; gap: 8px; padding: 2px 0; }
        .row span:last-child { color: var(--dim); }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); }
        .stat { background: var(--panel); padding: 8px 10px; }
        .stat b { display: block; font-size: 17px; font-weight: 600; color: var(--accent); }
        .stat i { font-style: normal; font-size: 9px; color: var(--faint);
                  letter-spacing: .8px; text-transform: uppercase; }
        .bar { height: 6px; background: var(--line); display: flex; overflow: hidden; }
        .bar div { height: 100%; }
        main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
        header {
          height: 46px; flex: 0 0 46px; border-bottom: 1px solid var(--line);
          display: flex; align-items: center; padding: 0 16px; gap: 14px;
        }
        .tabs { display: flex; gap: 2px; }
        .tab {
          padding: 5px 14px; background: none; border: 1px solid transparent;
          color: var(--dim); font: inherit; cursor: pointer;
        }
        .tab.on { color: var(--accent); border-color: var(--accent-dim); background: #0d2b31; }
        .pill {
          margin-left: auto; display: flex; gap: 14px; color: var(--dim); font-size: 11px;
        }
        .pill b { color: var(--text); font-weight: 600; }
        .view { flex: 1; overflow-y: auto; padding: 20px; display: none; }
        .view.on { display: block; }
        .msg { max-width: 780px; margin: 0 0 22px; }
        .who { font-size: 9px; letter-spacing: 1.2px; color: var(--faint); margin-bottom: 6px; }
        .msg.a .who { color: var(--accent); }
        .body { white-space: pre-wrap; word-wrap: break-word; }
        details { margin-top: 10px; border-left: 2px solid var(--line); padding-left: 10px; }
        summary { cursor: pointer; color: var(--faint); font-size: 11px; }
        details .body { color: var(--dim); font-size: 12px; margin-top: 6px; }
        footer { flex: 0 0 auto; border-top: 1px solid var(--line); padding: 12px 16px; }
        .composer { display: flex; gap: 10px; max-width: 780px; }
        textarea {
          flex: 1; background: #0c1117; border: 1px solid var(--line); color: var(--text);
          font: inherit; padding: 9px 11px; resize: none; height: 40px; outline: none;
        }
        textarea:focus { border-color: var(--accent-dim); }
        button.send {
          background: var(--accent-dim); border: none; color: var(--text);
          padding: 0 18px; cursor: pointer; font: inherit;
        }
        button.send:disabled { opacity: .35; cursor: default; }
        button.send.ghost { background: none; border: 1px solid var(--line);
                            color: var(--dim); }
        /* The grid needs an explicit height: 128 rows of 1fr inside an
           auto-height column collapse to nothing. */
        #grid { display: grid; gap: 1px; grid-template-columns: repeat(%LAYERS%, 1fr);
                background: var(--line); border: 1px solid var(--line);
                height: min(64vh, 640px); }
        .col { background: var(--bg); display: grid; gap: 1px;
               grid-template-rows: repeat(%EXPERTS%, 1fr); height: 100%; }
        .cell { background: #141b24; min-height: 1px; }
        .swatch.cell { min-height: 0; }
        .cell.hot { background: var(--accent); }
        .cell.warm { background: var(--trail); }
        .axis { display: flex; justify-content: space-between; color: var(--faint);
                font-size: 10px; margin-top: 6px; }
        .cut { color: var(--dim); font-size: 12px; margin-top: 8px;
               border-left: 2px solid var(--accent-dim); padding-left: 9px; }
        .legend { display: flex; gap: 16px; margin-bottom: 12px;
                  color: var(--dim); font-size: 11px; align-items: center; }
        .swatch { display: inline-block; width: 9px; height: 9px; margin-right: 5px;
                  vertical-align: -1px; }
        .empty { color: var(--faint); text-align: center; margin-top: 80px; }
        /* A grouped <select> rather than a list of buttons: it stays one line
           whether there are three models or thirty, and it is keyboard
           navigable for free. */
        .picker { display: flex; align-items: center; gap: 8px; }
        .picker > span {
          font-size: 9px; letter-spacing: 1.2px; text-transform: uppercase;
          color: var(--faint);
        }
        #model {
          appearance: none; background: #0c1117 no-repeat right 8px center/8px 5px;
          background-image: linear-gradient(45deg, transparent 50%, var(--dim) 50%),
                            linear-gradient(135deg, var(--dim) 50%, transparent 50%);
          background-position: right 13px center, right 8px center;
          background-size: 5px 5px, 5px 5px;
          border: 1px solid var(--line); color: var(--text); font: inherit;
          font-size: 12px; padding: 5px 28px 5px 10px; cursor: pointer; outline: none;
          max-width: 340px;
        }
        #model:hover, #model:focus { border-color: var(--accent-dim); }
        #model:disabled { opacity: .5; cursor: wait; }
        #install-progress {
          font-size: 11px; color: var(--dim); display: none; align-items: center; gap: 8px;
        }
        #install-progress .bar { width: 90px; height: 3px; background: var(--line); }
        #install-progress .bar div { height: 100%; background: var(--accent); width: 0 }
        #sky { width: 100%; height: min(66vh, 660px); display: block; cursor: grab;
               background: #06090d; border: 1px solid var(--line); }
        #sky:active { cursor: grabbing; }
        .sky-wrap { position: relative; }
        /* Sits over the canvas, so it can hold a button. */
        #range-empty {
          position: absolute; inset: 0; display: none;
          flex-direction: column; align-items: center; justify-content: center;
          gap: 12px; text-align: center; padding: 24px;
        }
        #range-empty b { font-size: 15px; }
        #range-empty p { color: var(--dim); font-size: 12px; max-width: 46ch;
                         line-height: 1.55; margin: 0; }
        #range-empty code { color: var(--faint); font-size: 11px; }
        #range-empty button {
          background: var(--accent); color: #04252b; border: none;
          font: inherit; font-size: 12px; font-weight: 600;
          padding: 8px 16px; cursor: pointer;
        }
        #range-empty button:disabled { background: var(--accent-dim);
                                       color: var(--dim); cursor: wait; }
        #range-progress { display: none; align-items: center; gap: 10px;
                          font-size: 11px; color: var(--dim); }
        #range-progress .bar { width: 160px; height: 3px; background: var(--line); }
        #range-progress .bar div { height: 100%; background: var(--accent); width: 0 }
        #range-legend { flex-wrap: wrap; gap: 10px 16px; }
        #range-legend span { display: flex; align-items: center; gap: 5px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        #tip { position: fixed; pointer-events: none; background: var(--panel);
               border: 1px solid var(--line); padding: 6px 9px; font-size: 11px;
               display: none; z-index: 10; white-space: pre; }
        </style>
        </head>
        <body>
        <aside>
          <div class="brand"><h1>Godwit</h1><span>MAX RANGE, MIN PAYLOAD</span></div>

          <section>
            <h2>Model</h2>
            <div class="row"><span>\(model)</span></div>
            <div class="row"><span>layers</span><span>\(layers)</span></div>
            <div class="row"><span>experts</span><span>\(experts) · top-\(topK)</span></div>
          </section>

          <section>
            <h2>Residency</h2>
            <div class="bar">
              <div style="background:var(--accent);width:%TRUNKPCT%%"></div>
              <div style="background:var(--accent-dim);width:%CACHEPCT%%"></div>
              <div style="background:var(--line);flex:1"></div>
            </div>
            <div class="row" style="margin-top:6px">
              <span><i class="swatch" style="background:var(--accent)"></i>trunk</span>
              <span>\(String(format: "%.2f", trunkGiB)) GiB</span>
            </div>
            <div class="row">
              <span><i class="swatch" style="background:var(--accent-dim)"></i>experts</span>
              <span>\(slots) slots · \(String(format: "%.2f", cacheGiB)) GiB</span>
            </div>
            <div class="row">
              <span><i class="swatch" style="background:var(--line)"></i>on disk</span>
              <span id="ondisk">streamed</span>
            </div>
          </section>

          <section>
            <h2>Last turn</h2>
            <div class="stat-grid">
              <div class="stat"><b id="s-rate">—</b><i>tok/s</i></div>
              <div class="stat"><b id="s-ttft">—</b><i>ttft</i></div>
              <div class="stat"><b id="s-tokens">—</b><i>generated</i></div>
              <div class="stat"><b id="s-hit">—</b><i>cache hit</i></div>
            </div>
          </section>

          <section>
            <h2>This turn read</h2>
            <div class="row"><span id="s-read">0.00 GiB</span><span id="s-misses">0 misses</span></div>
          </section>
        </aside>

        <main>
          <header>
            <label class="picker">
              <span>model</span>
              <select id="model"></select>
            </label>
            <div class="tabs">
              <button class="tab on" data-view="chat">Chat</button>
              <button class="tab" data-view="experts">Experts</button>
              <button class="tab" data-view="range">Range</button>
            </div>
            <div class="pill">
              <span id="install-progress"><span class="bar"><div></div></span>
                <b id="install-text"></b></span>
              <span>status <b id="status">idle</b></span>
              <span>layer <b id="layer">—</b></span>
            </div>
          </header>

          <div class="view on" id="v-chat"></div>

          <div class="view" id="v-experts">
            <div class="legend">
              <span><i class="swatch cell hot"></i>routed this token</span>
              <span><i class="swatch cell warm"></i>routed recently</span>
              <span>columns = layers 0-\(layers - 1), rows = experts 0-\(experts - 1)</span>
            </div>
            <div id="grid"></div>
            <div class="axis"><span>layer 0</span><span>layer \(layers - 1)</span></div>
          </div>

          <div class="view" id="v-range">
            <div class="legend" id="range-legend"></div>
            <div class="sky-wrap">
              <canvas id="sky"></canvas>
              <div id="range-empty"></div>
            </div>
            <div class="axis">
              <span id="range-note">loading…</span>
              <span>drag to spin · scroll to zoom</span>
            </div>
          </div>

          <footer>
            <div class="composer">
              <textarea id="input" placeholder="Ask something…"></textarea>
              <button class="send" id="send">Send</button>
              <button class="send ghost" id="new-chat"
                      title="Clear the conversation">New</button>
            </div>
          </footer>
        </main>

        <script>
        const LAYERS = \(layers), EXPERTS = \(experts);
        const $ = id => document.getElementById(id);

        document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
          document.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
          document.querySelectorAll('.view').forEach(x => x.classList.remove('on'));
          t.classList.add('on');
          $('v-' + t.dataset.view).classList.add('on');
          // The canvas has zero size while its tab is hidden, so anything drawn
          // before it is shown goes nowhere. Redraw on activation.
          if (t.dataset.view === 'range' && typeof drawRange === 'function') drawRange();
        });

        // 36 x 128 cells. Built once and mutated by class, never rebuilt —
        // touching 4,608 nodes per token is fine, recreating them is not.
        const cells = [];
        const grid = $('grid');
        for (let l = 0; l < LAYERS; l++) {
          const col = document.createElement('div');
          col.className = 'col';
          const column = [];
          for (let e = 0; e < EXPERTS; e++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            col.appendChild(cell);
            column.push(cell);
          }
          grid.appendChild(col);
          cells.push(column);
        }

        function paintRouting(layer, experts) {
          const column = cells[layer];
          for (const cell of column) {
            if (cell.classList.contains('hot')) {
              cell.classList.remove('hot');
              cell.classList.add('warm');
            }
          }
          for (const e of experts) if (column[e]) column[e].className = 'cell hot';
        }

        function addMessage(role, text) {
          const wrap = document.createElement('div');
          wrap.className = 'msg ' + (role === 'assistant' ? 'a' : 'u');
          wrap.innerHTML = '<div class="who">' + (role === 'assistant' ? 'GODWIT' : 'YOU')
            + '</div><div class="body"></div>';
          wrap.querySelector('.body').textContent = text;
          $('v-chat').appendChild(wrap);
          $('v-chat').scrollTop = $('v-chat').scrollHeight;
          return wrap;
        }

        // Clearing is a server-side operation: the session holds the KV cache
        // the conversation warmed, and dropping only the transcript would leave
        // the model still answering in the old context.
        $('new-chat').onclick = async () => {
          if (busy) return;
          await fetch('/api/chat/reset');
          $('v-chat').innerHTML = '';
          $('s-ttft').parentNode.title = '';
          $('status').textContent = 'idle';
          $('input').focus();
        };

        let busy = false;
        async function ask() {
          const text = $('input').value.trim();
          if (!text || busy) return;
          busy = true;
          $('send').disabled = true;
          $('input').value = '';
          addMessage('user', text);

          const reply = addMessage('assistant', '');
          const body = reply.querySelector('.body');
          let think = null;          // <details> holding the reasoning channel
          $('status').textContent = 'prefill';

          const source = new EventSource('/api/chat?q=' + encodeURIComponent(text));

          source.addEventListener('token', e => {
            const d = JSON.parse(e.data);
            body.textContent = d.text;
            // Once the answer begins, the reasoning stops being the interesting
            // thing on screen and folds itself away.
            if (d.text && think && think.open) think.open = false;
            $('v-chat').scrollTop = $('v-chat').scrollHeight;
          });
          source.addEventListener('routing', e => {
            const d = JSON.parse(e.data);
            paintRouting(d.layer, d.experts);
            $('layer').textContent = d.layer;
          });
          source.addEventListener('stats', e => {
            const d = JSON.parse(e.data);
            $('status').textContent = d.stage;
            if (d.rate !== undefined) $('s-rate').textContent = d.rate.toFixed(2);
            if (d.ttft !== undefined) {
              $('s-ttft').textContent = d.ttft.toFixed(1) + 's';
              // A follow-up prefills only its own new tokens, so say so —
              // otherwise a suddenly faster turn just looks like noise.
              $('s-ttft').parentNode.title = d.reused
                ? 'prefilled ' + (d.prompt - d.reused) + ' of ' + d.prompt
                  + ' tokens · ' + d.reused + ' reused from the conversation'
                : 'prefilled all ' + (d.prompt || 0) + ' prompt tokens';
            }
            if (d.tokens !== undefined) $('s-tokens').textContent = d.tokens;
            if (d.hit !== undefined) $('s-hit').textContent = (d.hit * 100).toFixed(0) + '%';
            if (d.read !== undefined) $('s-read').textContent = d.read.toFixed(2) + ' GiB';
            if (d.misses !== undefined) $('s-misses').textContent = d.misses + ' misses';
          });
          source.addEventListener('analysis', e => {
            const text = JSON.parse(e.data).text;
            if (!text) return;
            if (!think) {
              think = document.createElement('details');
              think.innerHTML = '<summary>reasoning</summary><div class="body"></div>';
              // Open while it is the only output there is, so a long silent
              // think reads as work rather than as a hang.
              think.open = !body.textContent;
              reply.insertBefore(think, body);
            }
            think.querySelector('.body').textContent = text;
            if (think.open) $('v-chat').scrollTop = $('v-chat').scrollHeight;
          });
          source.addEventListener('done', e => {
            const d = e.data ? JSON.parse(e.data) : {};
            if (d.answered === false) {
              // Leave the reasoning open — collapsing it would leave nothing
              // on screen at all.
              const note = document.createElement('div');
              note.className = 'cut';
              note.textContent = d.truncated
                ? 'Stopped at the ' + d.limit
                  + '-token limit while still reasoning, before reaching an answer.'
                : 'Ended without producing an answer.';
              reply.appendChild(note);
            } else if (think) {
              think.open = false;
            }
            source.close();
            busy = false;
            $('send').disabled = false;
            $('status').textContent = 'idle';
          });
          source.onerror = () => {
            source.close();
            busy = false;
            $('send').disabled = false;
            $('status').textContent = 'idle';
          };
        }


        // ---- Range: experts placed by measured routing affinity ----
        // A range map is the ornithologist's chart of where a species is
        // found; this is the same for experts, over topic space.
        // Colours are categorical and deliberately distinguishable on black;
        // they carry meaning (topic), so they must not be a gradient.
        const TOPIC_COLOURS = {
          python:'#60a5fa', sql:'#a3e635', math:'#fbbf24', poetry:'#f472b6',
          legal:'#c084fc', medical:'#34d399', chinese:'#fb7185', japanese:'#fb923c',
          russian:'#818cf8', json:'#2dd4bf', chat:'#94a3b8', history:'#d6a77a'
        };
        let rangeMap = null, spin = { x: -0.28, y: 0.7 }, zoom = 1.7, dragging = null;
        const sky = document.getElementById('sky');
        const tip = document.getElementById('tip');

        function project(p, w, h) {
          const cy = Math.cos(spin.y), sy = Math.sin(spin.y);
          const cx = Math.cos(spin.x), sx = Math.sin(spin.x);
          // Centre on the cloud, not the origin: the generalist core sits well
          // off zero and would otherwise push everything into a corner.
          const px = p.x - rangeMap.centre.x, py = p.y - rangeMap.centre.y,
                pz = p.z - rangeMap.centre.z;
          let x = px * cy - pz * sy;
          let z = px * sy + pz * cy;
          let y = py * cx - z * sx;
          z = py * sx + z * cx;
          const scale = Math.min(w, h) * 0.40 * zoom / (rangeMap.spread || 1);
          const depth = 1 / (1 + z * 0.35 / (rangeMap.spread || 1));
          return { sx: w / 2 + x * scale * depth, sy: h / 2 + y * scale * depth,
                   depth: depth, z: z };
        }

        // Shown over the canvas when the loaded model has no range map.
        function showRangeEmpty(state) {
          const box = $('range-empty');
          box.style.display = 'flex';
          if (state === 'building') return;
          box.innerHTML =
            '<b>No range map for this model.</b>'
            + '<p>It is measured rather than shipped. Probing the router with '
            + 'twenty-four samples takes a few minutes, and the model cannot '
            + 'answer while it runs — there is one GPU queue and one set of '
            + 'expert slots.</p>'
            + '<button id="build-range">Build range map</button>'
            + '<div id="range-progress"><div class="bar"><div></div></div>'
            + '<span></span></div>'
            + '<code>or: godwit range --model ' + missingDir
            + ' -o ' + missingDir + '/range.json</code>';
          $('build-range').onclick = buildRange;
        }

        function buildRange() {
          const box = $('range-empty');
          const button = $('build-range');
          const progress = $('range-progress');
          const fill = progress.querySelector('.bar div');
          const label = progress.querySelector('span');
          button.disabled = true;
          button.textContent = 'Probing…';
          progress.style.display = 'flex';
          $('status').textContent = 'probing';

          const source = new EventSource('/api/range/build');
          source.addEventListener('progress', e => {
            const d = JSON.parse(e.data);
            fill.style.width = (d.done / Math.max(d.total, 1) * 100) + '%';
            label.textContent = d.done + '/' + d.total + '  ' + d.topic;
          });
          source.addEventListener('done', e => {
            source.close();
            $('status').textContent = 'idle';
            // The map is on disk now, so the ordinary load path can have it.
            fetch('/api/range').then(r => r.json()).then(data => {
              box.style.display = 'none';
              missingDir = null;
              adoptRangeMap(data);
            });
          });
          source.addEventListener('error', e => {
            source.close();
            $('status').textContent = 'idle';
            button.disabled = false;
            button.textContent = 'Build range map';
            label.textContent = (JSON.parse(e.data || '{}').message) || 'failed';
          });
          source.onerror = () => {
            source.close();
            $('status').textContent = 'idle';
            button.disabled = false;
            button.textContent = 'Build range map';
          };
        }

        function drawRange() {
          if (!rangeMap && !missingDir) return;
          const dpr = window.devicePixelRatio || 1;
          const w = sky.clientWidth, h = sky.clientHeight;
          sky.width = w * dpr; sky.height = h * dpr;
          const g = sky.getContext('2d');
          g.scale(dpr, dpr);
          g.fillStyle = '#06090d'; g.fillRect(0, 0, w, h);

          // The empty state is DOM rather than canvas: it carries a button, and
          // a canvas cannot. It also cannot be drawn into a zero-sized canvas
          // by accident, which the drawn version had to be careful about.
          if (!rangeMap) { showRangeEmpty(); return; }

          // Far points first so near ones sit on top.
          const drawn = rangeMap.points.map(p => ({ p: p, v: project(p, w, h) }))
                                   .sort((a, b) => b.v.z - a.v.z);
          for (const d of drawn) {
            const spec = d.p.specialisation;
            const sure = d.p.confident !== false;
            const size = (0.7 + spec * 2.6) * d.v.depth * zoom * (sure ? 1 : 0.55);
            // Too few activations to read a topic into: still plotted, because
            // it is real routing, but not dressed up as a finding.
            g.globalAlpha = (0.25 + spec * 0.75) * (sure ? 1 : 0.3);
            g.fillStyle = TOPIC_COLOURS[d.p.topic] || '#64748b';
            g.beginPath();
            g.arc(d.v.sx, d.v.sy, Math.max(0.5, size), 0, 6.2832);
            g.fill();
          }

          // Topic labels at the centroid of each cluster's specialists.
          g.globalAlpha = 1;
          g.font = '11px ui-monospace, Menlo, monospace';
          for (const topic of rangeMap.topics) {
            const members = rangeMap.points.filter(p => p.topic === topic
                                                    && p.confident !== false
                                                    && p.specialisation > 0.45);
            if (members.length < 12) continue;
            let cx = 0, cy = 0, cz = 0;
            for (const m of members) { cx += m.x; cy += m.y; cz += m.z; }
            const v = project({ x: cx / members.length, y: cy / members.length,
                                z: cz / members.length }, w, h);
            // A dark plate behind the label: the clusters overlap near the
            // core and bare text on top of points is unreadable.
            const width = g.measureText(topic).width;
            g.fillStyle = 'rgba(6,9,13,0.86)';
            g.fillRect(v.sx + 4, v.sy - 14, width + 6, 15);
            g.fillStyle = TOPIC_COLOURS[topic] || '#94a3b8';
            g.fillText(topic, v.sx + 7, v.sy - 3);
            g.beginPath(); g.arc(v.sx, v.sy, 2.5, 0, 6.2832); g.fill();
          }
        }

        sky.addEventListener('mousedown', e => {
          dragging = { x: e.clientX, y: e.clientY, sx: spin.x, sy: spin.y };
        });
        window.addEventListener('mouseup', () => dragging = null);
        window.addEventListener('mousemove', e => {
          if (!dragging) return;
          spin.y = dragging.sy + (e.clientX - dragging.x) * 0.006;
          spin.x = dragging.sx + (e.clientY - dragging.y) * 0.006;
          drawRange();
        });
        sky.addEventListener('wheel', e => {
          e.preventDefault();
          zoom = Math.max(0.4, Math.min(6, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
          drawRange();
        }, { passive: false });

        sky.addEventListener('mousemove', e => {
          if (!rangeMap || dragging) { tip.style.display = 'none'; return; }
          const rect = sky.getBoundingClientRect();
          const mx = e.clientX - rect.left, my = e.clientY - rect.top;
          let best = null, bestDist = 64;
          for (const p of rangeMap.points) {
            const v = project(p, sky.clientWidth, sky.clientHeight);
            const d = (v.sx - mx) ** 2 + (v.sy - my) ** 2;
            if (d < bestDist) { bestDist = d; best = p; }
          }
          if (!best) { tip.style.display = 'none'; return; }
          tip.textContent = 'layer ' + best.layer + ' · expert ' + best.expert
            + '\\n' + best.topic + '  ' + (best.specialisation * 100).toFixed(0)
            + '% concentrated\\n' + best.activations + ' activations';
          tip.style.display = 'block';
          tip.style.left = (e.clientX + 14) + 'px';
          tip.style.top = (e.clientY + 14) + 'px';
        });
        sky.addEventListener('mouseleave', () => tip.style.display = 'none');

        // Adopting a map is shared between the page load and the build button,
        // so a freshly probed map renders exactly like a loaded one.
        function adoptRangeMap(data) {
          const note = $('range-note');
          rangeMap = data;
          let cx = 0, cy = 0, cz = 0;
          for (const p of data.points) { cx += p.x; cy += p.y; cz += p.z; }
          const n = data.points.length;
          rangeMap.centre = { x: cx / n, y: cy / n, z: cz / n };
          let spread = 0;
          for (const p of data.points) {
            spread = Math.max(spread, Math.abs(p.x - rangeMap.centre.x),
                              Math.abs(p.y - rangeMap.centre.y),
                              Math.abs(p.z - rangeMap.centre.z));
          }
          rangeMap.spread = spread || 1;

          const legend = $('range-legend');
          legend.innerHTML = '';
          for (const topic of data.topics) {
            // Counts are of experts that cleared the activation bar. This is
            // the number the legend used to overstate.
            const count = data.points.filter(p => p.topic === topic
                                              && p.confident !== false
                                              && p.specialisation > 0.35).length;
            const el = document.createElement('span');
            el.innerHTML = '<i class="dot" style="background:'
              + (TOPIC_COLOURS[topic] || '#64748b') + '"></i>' + topic + ' ' + count;
            legend.appendChild(el);
          }
          const v = data.variance || [];
          note.textContent = data.points.length + ' experts · axes explain '
            + v.map(x => (x * 100).toFixed(0) + '%').join(' / ') + ' of variance';
          drawRange();
        }

        fetch('/api/range').then(r => r.ok ? r.json() : null).then(data => {
          if (!data || !data.points || !data.points.length) {
            $('range-note').textContent = 'no range map for this model';
            missingDir = picker.dataset.current || '<dir>';
            showRangeEmpty();
            return;
          }
          adoptRangeMap(data);
        });
        window.addEventListener('resize', drawRange);


        // Set once we know a range map is absent, to the directory whose map
        // is missing. Null while a map is present or still loading.
        let missingDir = null;

        // ---- Models ----
        // One <select> holds both what is installed and what can be installed,
        // in separate groups. Choosing an install entry starts the download;
        // choosing an installed one swaps the loaded model.
        let switching = false;
        const picker = $('model');

        function fmtGiB(b) { return (b / 1073741824).toFixed(1) + ' GiB'; }

        async function refreshModels() {
          const d = await (await fetch('/api/models')).json();
          picker.innerHTML = '';

          if (d.installed.length) {
            const g = document.createElement('optgroup');
            g.label = 'installed';
            for (const m of d.installed) {
              const o = document.createElement('option');
              o.value = m.name;
              o.textContent = m.model.split('/').pop()
                + '  ·  ' + m.layers + 'L × ' + m.experts + 'E  ·  ' + fmtGiB(m.bytes)
                + (m.complete ? '' : '  · partial');
              o.selected = m.name === d.active;
              o.disabled = !m.complete;
              g.appendChild(o);
            }
            picker.appendChild(g);
          }

          // Only a complete install counts as having the model — a partial
          // one should still offer the download that would finish the job.
          const have = new Set(d.installed.filter(m => m.complete).map(m => m.model));
          const missing = d.available.filter(a => !have.has(a.id));
          if (missing.length) {
            const g = document.createElement('optgroup');
            g.label = 'install';
            for (const a of missing) {
              const o = document.createElement('option');
              o.value = '+' + a.id;
              o.textContent = '↓ ' + a.title + '  ·  ' + fmtGiB(a.bytes);
              o.title = a.note;
              g.appendChild(o);
            }
            picker.appendChild(g);
          }

          if (!d.active) {
            const o = document.createElement('option');
            o.textContent = 'none loaded'; o.value = ''; o.selected = true;
            picker.insertBefore(o, picker.firstChild);
          }
          picker.dataset.current = d.active || '';
          if (missingDir !== null && d.active) { missingDir = d.active; drawRange(); }
          if (d.installing) startWatching(d.installing);
        }

        picker.onchange = async () => {
          const v = picker.value;
          if (v.startsWith('+')) { installModel(v.slice(1)); return; }
          if (!v || v === picker.dataset.current) return;
          if (switching || busy) { picker.value = picker.dataset.current; return; }

          switching = true; picker.disabled = true;
          $('status').textContent = 'loading';
          const r = await (await fetch('/api/select?name=' + encodeURIComponent(v))).json();
          switching = false; picker.disabled = false;
          if (r.error) { alert(r.error); await refreshModels(); return; }
          // The expert grid is sized by the model's layer and expert counts, so
          // a switch rebuilds the page rather than patching it.
          location.reload();
        };

        function startWatching(id) {
          const box = $('install-progress'), text = $('install-text');
          const fill = box.querySelector('.bar div');
          box.style.display = 'flex';
          const size = 1;
          const src = new EventSource('/api/install?id=' + encodeURIComponent(id));
          src.addEventListener('progress', e => {
            const p = JSON.parse(e.data);
            text.textContent = p.stage + ' ' + p.done + '/' + p.total
              + '  ' + fmtGiB(p.bytes);
            fill.style.width = Math.min(100, p.done / Math.max(p.total, 1) * 100) + '%';
          });
          src.addEventListener('done', () => {
            src.close(); box.style.display = 'none'; refreshModels();
          });
          src.addEventListener('error', e => {
            src.close(); box.style.display = 'none';
            alert((JSON.parse(e.data || '{}').message) || 'install failed');
            refreshModels();
          });
          src.onerror = () => { src.close(); box.style.display = 'none'; };
        }

        function installModel(id) {
          picker.value = picker.dataset.current;
          startWatching(id);
        }

        refreshModels();

        $('send').onclick = ask;
        $('input').addEventListener('keydown', e => {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
        });
        </script>
        <div id="tip"></div>
        </body>
        </html>
        """
        .replacingOccurrences(of: "%LAYERS%", with: String(layers))
        .replacingOccurrences(of: "%EXPERTS%", with: String(experts))
        .replacingOccurrences(of: "%TRUNKPCT%",
                              with: String(format: "%.1f", trunkGiB / 16 * 100))
        .replacingOccurrences(of: "%CACHEPCT%",
                              with: String(format: "%.1f", cacheGiB / 16 * 100))
    }
}
