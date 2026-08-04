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
        <title>godwit</title>
        <style>
        :root {
          --bg: #0b0c0e; --panel: #131519; --line: #23262c;
          --text: #e8e6e3; --dim: #8b8783; --faint: #55524f;
          --accent: #d98b4a; --accent-dim: #6b4526; --good: #7fb069;
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
        .tab.on { color: var(--accent); border-color: var(--accent-dim); background: #1a1512; }
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
          flex: 1; background: #0e1013; border: 1px solid var(--line); color: var(--text);
          font: inherit; padding: 9px 11px; resize: none; height: 40px; outline: none;
        }
        textarea:focus { border-color: var(--accent-dim); }
        button.send {
          background: var(--accent-dim); border: none; color: var(--text);
          padding: 0 18px; cursor: pointer; font: inherit;
        }
        button.send:disabled { opacity: .35; cursor: default; }
        /* The grid needs an explicit height: 128 rows of 1fr inside an
           auto-height column collapse to nothing. */
        #grid { display: grid; gap: 1px; grid-template-columns: repeat(%LAYERS%, 1fr);
                background: var(--line); border: 1px solid var(--line);
                height: min(64vh, 640px); }
        .col { background: var(--bg); display: grid; gap: 1px;
               grid-template-rows: repeat(%EXPERTS%, 1fr); height: 100%; }
        .cell { background: #16181c; min-height: 1px; }
        .swatch.cell { min-height: 0; }
        .cell.hot { background: var(--accent); }
        .cell.warm { background: var(--accent-dim); }
        .axis { display: flex; justify-content: space-between; color: var(--faint);
                font-size: 10px; margin-top: 6px; }
        .legend { display: flex; gap: 16px; margin-bottom: 12px;
                  color: var(--dim); font-size: 11px; align-items: center; }
        .swatch { display: inline-block; width: 9px; height: 9px; margin-right: 5px;
                  vertical-align: -1px; }
        .empty { color: var(--faint); text-align: center; margin-top: 80px; }
        </style>
        </head>
        <body>
        <aside>
          <div class="brand"><h1>godwit</h1><span>MAX RANGE, MIN PAYLOAD</span></div>

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
            <div class="tabs">
              <button class="tab on" data-view="chat">Chat</button>
              <button class="tab" data-view="experts">Experts</button>
            </div>
            <div class="pill">
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

          <footer>
            <div class="composer">
              <textarea id="input" placeholder="Ask something…"></textarea>
              <button class="send" id="send">Send</button>
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
          let analysis = null;
          $('status').textContent = 'prefill';

          const source = new EventSource('/api/chat?q=' + encodeURIComponent(text));

          source.addEventListener('token', e => {
            const d = JSON.parse(e.data);
            body.textContent = d.text;
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
            if (d.ttft !== undefined) $('s-ttft').textContent = d.ttft.toFixed(1) + 's';
            if (d.tokens !== undefined) $('s-tokens').textContent = d.tokens;
            if (d.hit !== undefined) $('s-hit').textContent = (d.hit * 100).toFixed(0) + '%';
            if (d.read !== undefined) $('s-read').textContent = d.read.toFixed(2) + ' GiB';
            if (d.misses !== undefined) $('s-misses').textContent = d.misses + ' misses';
          });
          source.addEventListener('analysis', e => {
            analysis = JSON.parse(e.data).text;
          });
          source.addEventListener('done', () => {
            if (analysis) {
              const d = document.createElement('details');
              d.innerHTML = '<summary>reasoning</summary><div class="body"></div>';
              d.querySelector('.body').textContent = analysis;
              reply.appendChild(d);
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

        $('send').onclick = ask;
        $('input').addEventListener('keydown', e => {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
        });
        </script>
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
