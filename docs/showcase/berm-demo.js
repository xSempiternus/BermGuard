/* BermGuard · demo del camino óptimo.
 *
 * Reproduce en el navegador el experimento del ADR 0006 sobre una escena sintética:
 * la cresta del pretil es un escalón de brillo de bajo contraste y las huellas de
 * neumático son tramos horizontales cortos, más contrastados, bajo ella. Los dos
 * métodos ven la misma respuesta de gradiente:
 *
 *   argmax    cada columna elige su máximo por separado y un filtro suaviza después.
 *   camino    programación dinámica: el recorrido de máxima respuesta que cruza
 *             todas las columnas con un salto vertical acotado.
 *
 * Es una versión reducida de bermguard/vision/berm/, no una copia: mismo principio,
 * a una escala que corre en cada movimiento de un slider.
 *
 * Uso: BermDemo.mount(elemento)
 */
(function (global) {
  'use strict';

  var W = 320, H = 150, BAND_TOP = 18, BAND_BOTTOM = Math.round(H * 0.82), MAX_STEP = 1;

  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(r) {
    var u = 1 - r(), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* Escena: terreno, cresta verdadera y huellas. Devuelve la imagen en gris. */
  function scene(p) {
    var r = rng(p.seed), img = new Float32Array(W * H), crest = new Float32Array(W);
    var ph = r() * 6.28;
    for (var x = 0; x < W; x++) {
      crest[x] = 62 + 9 * Math.sin(2 * Math.PI * x / W * 1.4 + ph) + 3.5 * Math.sin(2 * Math.PI * x / W * 4 + 1.3);
    }
    for (var y = 0; y < H; y++) for (x = 0; x < W; x++) {
      img[y * W + x] = 70 + 0.16 * y + (y > crest[x] ? p.contrast : 0);
    }
    var lo = 90, hi = BAND_BOTTOM - 4;
    for (var k = 0; k < p.tracks; k++) {
      var row = Math.floor(lo + r() * (hi - lo)), len = Math.floor(W * (0.1 + r() * 0.22));
      var x0 = Math.floor(r() * (W - len)), th = 2 + Math.floor(r() * 2);
      for (y = row; y < row + th; y++) for (x = x0; x < x0 + len; x++) img[y * W + x] += 30;
    }
    for (var i = 0; i < img.length; i++) img[i] += p.noise * gauss(r);
    return { img: img, crest: crest };
  }

  /* Desenfoque gaussiano aproximado con tres pasadas de caja, separable. */
  function blur(src, rad) {
    var a = Float32Array.from(src), b = new Float32Array(src.length);
    for (var pass = 0; pass < 3; pass++) {
      for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
        var s = 0, n = 0;
        for (var d = -rad; d <= rad; d++) { var xx = x + d; if (xx >= 0 && xx < W) { s += a[y * W + xx]; n++; } }
        b[y * W + x] = s / n;
      }
      for (y = 0; y < H; y++) for (x = 0; x < W; x++) {
        s = 0; n = 0;
        for (d = -rad; d <= rad; d++) { var yy = y + d; if (yy >= 0 && yy < H) { s += b[yy * W + x]; n++; } }
        a[y * W + x] = s / n;
      }
    }
    return a;
  }

  /* Gradiente vertical positivo: el terreno se aclara al bajar la cresta. */
  function response(img) {
    var s = blur(img, 2), g = new Float32Array(W * H);
    for (var y = 1; y < H - 1; y++) for (var x = 0; x < W; x++) {
      g[y * W + x] = Math.max(0, s[(y + 1) * W + x] - s[(y - 1) * W + x]);
    }
    return g;
  }

  function argmaxPath(g) {
    var raw = new Float32Array(W);
    for (var x = 0; x < W; x++) {
      var best = -1, by = BAND_TOP;
      for (var y = BAND_TOP; y < BAND_BOTTOM; y++) if (g[y * W + x] > best) { best = g[y * W + x]; by = y; }
      raw[x] = by;
    }
    // La reparación a posteriori del Método 2: mediana móvil. Atenúa los saltos,
    // pero no puede recuperar la elección correcta.
    var out = new Float32Array(W), win = 7, buf = [];
    for (x = 0; x < W; x++) {
      buf.length = 0;
      for (var d = -win; d <= win; d++) buf.push(raw[Math.min(W - 1, Math.max(0, x + d))]);
      buf.sort(function (a, b) { return a - b; });
      out[x] = buf[win];
    }
    return out;
  }

  function dpPath(g) {
    var rows = BAND_BOTTOM - BAND_TOP, acc = new Float32Array(rows * W);
    for (var y = 0; y < rows; y++) acc[y * W] = g[(y + BAND_TOP) * W];
    for (var x = 1; x < W; x++) for (y = 0; y < rows; y++) {
      var m = -1;
      for (var d = -MAX_STEP; d <= MAX_STEP; d++) {
        var yy = y + d; if (yy >= 0 && yy < rows && acc[yy * W + x - 1] > m) m = acc[yy * W + x - 1];
      }
      acc[y * W + x] = g[(y + BAND_TOP) * W + x] + m;
    }
    var path = new Float32Array(W), by = 0, best = -1;
    for (y = 0; y < rows; y++) if (acc[y * W + W - 1] > best) { best = acc[y * W + W - 1]; by = y; }
    path[W - 1] = by + BAND_TOP;
    for (x = W - 2; x >= 0; x--) {
      var b2 = -1, ny = by;
      for (d = -MAX_STEP; d <= MAX_STEP; d++) {
        yy = by + d; if (yy >= 0 && yy < rows && acc[yy * W + x] > b2) { b2 = acc[yy * W + x]; ny = yy; }
      }
      by = ny; path[x] = by + BAND_TOP;
    }
    return path;
  }

  function within(path, crest, tol) {
    var ok = 0;
    for (var x = 0; x < W; x++) if (Math.abs(path[x] - crest[x]) <= tol) ok++;
    return ok / W;
  }

  var CSS =
    '.bdm{font:13px/1.4 Tahoma,Verdana,sans-serif;color:#101010}' +
    '.bdm canvas{display:block;width:100%;height:auto;image-rendering:pixelated;background:#000;border:2px inset #dcdcdc}' +
    '.bdm .row{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:8px 0}' +
    '.bdm label{display:flex;align-items:center;gap:6px;font-size:12px}' +
    '.bdm input[type=range]{width:120px}' +
    '.bdm .score{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:6px}' +
    '.bdm .score div{padding:6px 8px;border:2px groove #dcdcdc;background:#fff}' +
    '.bdm .score b{display:block;font:400 30px/1 VT323,"Courier New",monospace}' +
    '.bdm .score span{font-size:11.5px;color:#4a4a4a}' +
    '.bdm .k{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle}' +
    '.bdm button{padding:4px 12px;font:inherit}' +
    '.bdm .avg{margin:8px 0 0;max-width:none;font-size:12.5px;color:#4a4a4a}';

  function mount(root, opts) {
    opts = opts || {};
    if (!document.getElementById('bdm-css')) {
      var st = document.createElement('style'); st.id = 'bdm-css'; st.textContent = CSS; document.head.appendChild(st);
    }
    root.classList.add('bdm');
    root.innerHTML =
      '<canvas width="' + W + '" height="' + H + '" role="img" aria-label="Escena sintética con la cresta verdadera y la estimada por cada método"></canvas>' +
      '<div class="row">' +
      '<label>Huellas <input type="range" min="0" max="14" value="4" data-k="tracks"><output>4</output></label>' +
      '<label>Contraste cresta <input type="range" min="6" max="24" value="14" data-k="contrast"><output>14</output></label>' +
      '<label>Ruido <input type="range" min="0" max="16" value="6" data-k="noise"><output>6</output></label>' +
      '<button type="button" class="' + (opts.buttonClass || '') + '">Otra escena</button></div>' +
      '<div class="row" style="font-size:12px"><span><i class="k" style="background:#ffd23a"></i>cresta real</span>' +
      '<span><i class="k" style="background:#ff4a3d"></i>argmax por columna + filtro</span>' +
      '<span><i class="k" style="background:#39ff6a"></i>camino óptimo</span></div>' +
      '<div class="score"><div><b data-s="am">–</b><span>argmax: columnas a ≤3 px de la cresta</span></div>' +
      '<div><b data-s="dp">–</b><span>camino óptimo: columnas a ≤3 px</span></div></div>' +
      '<p class="avg" data-s="avg" aria-live="polite"></p>';

    var cv = root.querySelector('canvas'), ctx = cv.getContext('2d');
    // Semilla 21: la mediana de la diferencia entre métodos en 40 escenas con estos
    // parámetros. Ni el mejor caso ni el peor.
    var p = { seed: 21, tracks: 4, contrast: 14, noise: 6 }, avgTimer = 0;

    // Una escena sola puede engañar en cualquier dirección: el promedio en 20 escenas
    // con los mismos parámetros es lo que dice cuál método es mejor.
    function average() {
      var a = 0, d = 0, n = 20;
      for (var k = 1; k <= n; k++) {
        var s = scene({ seed: 1000 + k, tracks: p.tracks, contrast: p.contrast, noise: p.noise });
        var g = response(s.img);
        a += within(argmaxPath(g), s.crest, 3); d += within(dpPath(g), s.crest, 3);
      }
      root.querySelector('[data-s=avg]').textContent =
        'Promedio en ' + n + ' escenas con estos parámetros: argmax ' + Math.round(a / n * 100) +
        ' %, camino óptimo ' + Math.round(d / n * 100) + ' %.';
    }

    function draw() {
      var s = scene(p), g = response(s.img), am = argmaxPath(g), dp = dpPath(g);
      var im = ctx.createImageData(W, H);
      for (var i = 0; i < W * H; i++) {
        var v = Math.max(0, Math.min(255, s.img[i]));
        // Paleta terrosa: el material es ocre, no gris.
        im.data[4 * i] = v * 1.05; im.data[4 * i + 1] = v * 0.86; im.data[4 * i + 2] = v * 0.62; im.data[4 * i + 3] = 255;
      }
      ctx.putImageData(im, 0, 0);
      ctx.fillStyle = 'rgba(0,0,40,.35)';
      ctx.fillRect(0, 0, W, BAND_TOP); ctx.fillRect(0, BAND_BOTTOM, W, H - BAND_BOTTOM);
      function line(path, color, dash) {
        ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 1.4; if (dash) ctx.setLineDash(dash);
        ctx.beginPath(); for (var x = 0; x < W; x++) { if (x) ctx.lineTo(x + 0.5, path[x]); else ctx.moveTo(0.5, path[0]); }
        ctx.stroke(); ctx.restore();
      }
      // La cresta real va encima: si un método la acierta, no debe taparla.
      line(am, '#ff4a3d'); line(dp, '#39ff6a'); line(s.crest, '#ffd23a', [3, 3]);
      root.querySelector('[data-s=am]').textContent = Math.round(within(am, s.crest, 3) * 100) + ' %';
      root.querySelector('[data-s=dp]').textContent = Math.round(within(dp, s.crest, 3) * 100) + ' %';
      clearTimeout(avgTimer); avgTimer = setTimeout(average, 120);
    }

    root.addEventListener('input', function (e) {
      var k = e.target.dataset && e.target.dataset.k; if (!k) return;
      p[k] = +e.target.value; e.target.nextElementSibling.textContent = e.target.value; draw();
    });
    root.querySelector('button').addEventListener('click', function () { p.seed = (p.seed * 48271 + 11) % 2147483647; draw(); });
    draw();
  }

  global.BermDemo = { mount: mount };
})(typeof window !== 'undefined' ? window : this);
