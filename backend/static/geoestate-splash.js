/*! GeoEstate launch screen, v1.0 · shared by every GeoEstate application (Geohub, PolyGIS, Land Registry, LandCheck, FarmScan, CRM…)
 *
 * The first thing on the page, before the application's own code has even downloaded, as desktop software shows its splash
 * window while it starts. It needs nothing else: no framework, no fonts, no network (only the app's /logo.png).
 *
 * Use: the first element in <body>, as a plain (blocking) script so it paints at once:
 *
 *   <script src="/geoestate-splash.js" data-product="PolyGIS" data-tagline="Web GIS for surveyors and planners"
 *           data-edition="" data-version="1.0.0" data-wait="app"></script>
 *
 *   data-product   the application's name (required)
 *   data-tagline   one line under the name
 *   data-edition   e.g. "Advanced"; can be set later with GeoEstateSplash.edition("Advanced")
 *   data-version   shown at the bottom
 *   data-accent    the product's colour (default GeoEstate green #3aa335)
 *   data-logo      default /logo.png
 *   data-min       how long the full launch lasts, in seconds (default 10, 1 to 20)
 *   data-wait      "app": stay until the application calls GeoEstateSplash.done() (at most 15 s more);
 *                  otherwise (default): leave when the page has loaded
 *
 * From the application:
 *   GeoEstateSplash.step("Checking your GeoEstate ID")       what it is doing (the bar follows the launch's time)
 *   GeoEstateSplash.done()                                     ready: the screen finishes its bar and fades away
 *
 * The full launch (data-min seconds, 10 by default) plays once per browser session ("cold start"): the bar fills over that
 * time while the steps of starting go by, and once the application is ready, "Skip" (or Esc, or a click outside the window)
 * ends it early. A reload in the same session shows a short one ("warm start"). Reduced-motion settings are respected.
 */
(function () {
  "use strict";
  if (window.GeoEstateSplash) return;
  var me = document.currentScript || {};
  var ds = me.dataset || {};
  var product = ds.product || document.title.split(/[–|-]/)[0].trim() || "GeoEstate";
  var accent = ds.accent || "#3aa335";
  var key = "ge-splash:" + product;
  var warm = false;
  try { warm = sessionStorage.getItem(key) === "1"; sessionStorage.setItem(key, "1"); } catch (e) { /* private window */ }
  var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // how long the full launch lasts (data-min, in seconds; 10 by default), like desktop software starting; a reload in the same
  // session is quick, and reduced motion quicker still
  var coldMs = Math.max(1, Math.min(20, parseFloat(ds.min) || 10)) * 1000;
  var minMs = still ? 300 : warm ? 450 : coldMs;
  var t0 = Date.now();
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };

  var css = [
    ".ges{position:fixed;inset:0;z-index:2147483000;display:grid;place-items:center;overflow:hidden;font-family:Inter,'Segoe UI',system-ui,-apple-system,Roboto,Arial,sans-serif;",
    "background:radial-gradient(1200px 700px at 50% 42%,#13284a 0%,#0b1830 55%,#060e1e 100%);color:#e8eef8;-webkit-font-smoothing:antialiased;",
    "transition:opacity .38s ease,visibility .38s;}",
    ".ges.is-out{opacity:0;visibility:hidden;}",
    // the application's own styles (headings, paragraphs, images) must not reach in
    ".ges *,.ges *::before,.ges *::after{box-sizing:border-box;}",
    ".ges h1.ges-name,.ges p.ges-tag{margin:0;padding:0;border:0;background:none;text-transform:none;}",
    ".ges img{max-width:none;}",
    ".ges-grid{position:absolute;inset:-20%;background-image:linear-gradient(rgba(120,170,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(120,170,255,.06) 1px,transparent 1px);",
    "background-size:56px 56px;transform:perspective(900px) rotateX(58deg) translateY(18%);animation:ges-drift 18s linear infinite;mask-image:radial-gradient(circle at 50% 40%,#000 0,transparent 70%);-webkit-mask-image:radial-gradient(circle at 50% 40%,#000 0,transparent 70%);}",
    "@keyframes ges-drift{to{background-position:0 560px,560px 0;}}",
    ".ges-win{position:relative;width:min(640px,calc(100vw - 32px));border-radius:14px;overflow:hidden;",
    "background:linear-gradient(160deg,#162d52 0%,#0f2142 48%,#0b1a35 100%);border:1px solid rgba(160,190,240,.16);",
    "box-shadow:0 30px 80px rgba(0,0,0,.55),0 2px 0 rgba(255,255,255,.04) inset;animation:ges-in .55s cubic-bezier(.2,.8,.2,1) both;}",
    ".ges.is-warm .ges-win{animation-duration:.2s;}",
    "@keyframes ges-in{from{opacity:0;transform:translateY(14px) scale(.97);}to{opacity:1;transform:none;}}",
    ".ges.is-out .ges-win{transform:scale(1.02);transition:transform .38s ease;}",
    ".ges-body{display:grid;grid-template-columns:228px 1fr;gap:8px;align-items:center;padding:30px 34px 12px 26px;min-height:236px;}",
    ".ges-art{position:relative;width:220px;height:220px;}",
    ".ges-art svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible;}",
    ".ges-glow{position:absolute;inset:18px;border-radius:50%;background:radial-gradient(circle,color-mix(in srgb,var(--ges-a) 28%,transparent) 0%,transparent 68%);filter:blur(6px);animation:ges-pulse 2.6s ease-in-out infinite;}",
    "@keyframes ges-pulse{50%{opacity:.55;transform:scale(.94);}}",
    ".ges-draw{stroke-dasharray:var(--l);stroke-dashoffset:var(--l);animation:ges-draw 1.1s cubic-bezier(.6,.1,.3,1) forwards;}",
    "@keyframes ges-draw{to{stroke-dashoffset:0;}}",
    ".ges-spin{transform-origin:110px 112px;animation:ges-spin 9s linear infinite;}",
    "@keyframes ges-spin{to{transform:rotate(360deg);}}",
    ".ges-meridians ellipse{animation:ges-draw 1.1s cubic-bezier(.6,.1,.3,1) forwards,ges-turn 4.5s linear 1.1s infinite;}",
    "@keyframes ges-turn{0%{rx:var(--r0);}50%{rx:var(--r1);}100%{rx:var(--r0);}}",
    ".ges-layer{opacity:0;animation:ges-stack .5s cubic-bezier(.2,.9,.3,1.2) forwards;}",
    "@keyframes ges-stack{from{opacity:0;transform:translateY(-26px);}to{opacity:1;transform:none;}}",
    ".ges-pin{opacity:0;animation:ges-drop .55s cubic-bezier(.3,1.4,.5,1) 1.55s forwards;}",
    "@keyframes ges-drop{from{opacity:0;transform:translateY(-34px);}to{opacity:1;transform:none;}}",
    ".ges-ring{transform-origin:150px 72px;opacity:0;animation:ges-ring 1.8s ease-out 2s infinite;}",
    "@keyframes ges-ring{0%{opacity:.8;transform:scale(.3);}100%{opacity:0;transform:scale(1.9);}}",
    ".ges-scan{transform-origin:110px 112px;animation:ges-spin 2.4s linear infinite;}",
    ".ges-txt{min-width:0;}",
    ".ges-brand{display:flex;align-items:center;gap:10px;margin-bottom:14px;}",
    ".ges-brand img{width:38px;height:38px;border-radius:9px;background:#fff;padding:3px;object-fit:contain;box-shadow:0 2px 8px rgba(0,0,0,.3);}",
    ".ges-brand span{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#9fb3d6;font-weight:600;}",
    ".ges h1.ges-name{font-family:inherit;font-size:44px;line-height:1.02;font-weight:700;letter-spacing:-.02em;color:#fff;margin:0;opacity:0;animation:ges-up .6s ease .25s forwards;}",
    ".ges h1.ges-name b{color:var(--ges-a);font-weight:700;}",
    ".ges p.ges-tag{font-family:inherit;margin:8px 0 0;font-size:14.5px;line-height:1.45;color:#b9c8e3;opacity:0;animation:ges-up .6s ease .45s forwards;}",
    ".ges-ed{display:inline-block;margin-top:12px;padding:3px 9px;border-radius:999px;font-size:11.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;",
    "color:var(--ges-a);background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);}",
    ".ges-ed:empty{display:none;}",
    "@keyframes ges-up{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}",
    ".ges-foot{padding:14px 26px 18px;border-top:1px solid rgba(160,190,240,.10);background:rgba(4,10,24,.25);}",
    ".ges-status{display:flex;justify-content:space-between;gap:12px;font-size:12.5px;color:#c9d6ee;margin-bottom:8px;min-height:18px;}",
    ".ges-status span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}",
    ".ges-status .ges-pct{color:#7f93b7;font-variant-numeric:tabular-nums;}",
    ".ges-bar{height:3px;border-radius:3px;background:rgba(255,255,255,.08);overflow:hidden;}",
    ".ges-bar i{display:block;height:100%;width:0;border-radius:3px;background:linear-gradient(90deg,var(--ges-a),#7fd36e);box-shadow:0 0 12px var(--ges-a);transition:width .45s ease;}",
    ".ges-skip{margin-left:8px;padding:0;border:0;background:none;font:inherit;color:#8fb6ff;cursor:pointer;opacity:0;transition:opacity .4s;}",
    ".ges-skip.is-on{opacity:1;}.ges-skip:hover{text-decoration:underline;}",
    ".ges-legal{display:flex;justify-content:space-between;gap:12px;margin-top:12px;font-size:11px;color:#6f84a8;}",
    "@media (max-width:560px){.ges-body{grid-template-columns:1fr;justify-items:center;text-align:center;padding:24px 20px 8px;}",
    ".ges-art{width:150px;height:150px;}.ges-brand{justify-content:center;}.ges h1.ges-name{font-size:36px;}.ges-legal{flex-direction:column;gap:2px;align-items:center;}}",
    ".ges.is-warm *,.ges.is-still *{animation-duration:.01s!important;animation-delay:0s!important;animation-iteration-count:1!important;}",
    ".ges.is-still .ges-grid{animation:none;}",
  ].join("");

  // the art: a wireframe globe drawing itself, map layers stacking onto it, a location pin dropping with a ripple
  var art =
    '<div class="ges-glow"></div>' +
    '<svg viewBox="0 0 220 220" aria-hidden="true" fill="none" stroke-linecap="round">' +
    '<circle class="ges-draw" style="--l:520" cx="110" cy="112" r="80" stroke="#2a4f86" stroke-width="2"/>' +
    '<g class="ges-meridians" stroke="var(--ges-a)" stroke-opacity=".75" stroke-width="1.4">' +
    '<ellipse class="ges-draw" style="--l:420;--r0:22px;--r1:58px" cx="110" cy="112" rx="40" ry="80"/>' +
    '<ellipse class="ges-draw" style="--l:420;--r0:58px;--r1:22px;animation-delay:.12s" cx="110" cy="112" rx="40" ry="80"/></g>' +
    '<g stroke="#5f8fd0" stroke-opacity=".55" stroke-width="1.2">' +
    '<path class="ges-draw" style="--l:160;animation-delay:.2s" d="M34 112h152"/>' +
    '<path class="ges-draw" style="--l:140;animation-delay:.3s" d="M45 80q65 -16 130 0"/>' +
    '<path class="ges-draw" style="--l:140;animation-delay:.36s" d="M45 144q65 16 130 0"/>' +
    '<path class="ges-draw" style="--l:180;animation-delay:.1s" d="M110 32v160"/></g>' +
    '<g class="ges-scan"><path d="M110 112 L110 30 A82 82 0 0 1 168 54 Z" fill="url(#ges-sweep)" stroke="none"/></g>' +
    '<g class="ges-spin"><circle cx="110" cy="112" r="96" stroke="#3b6bb0" stroke-opacity=".35" stroke-width="1" stroke-dasharray="2 7"/></g>' +
    '<g transform="translate(118 44)">' +
    '<path class="ges-layer" style="animation-delay:.9s" d="M0 46 L34 30 L68 46 L34 62 Z" fill="#0f2c55" stroke="#4f86d6" stroke-width="1.5"/>' +
    '<path class="ges-layer" style="animation-delay:1.1s" d="M0 36 L34 20 L68 36 L34 52 Z" fill="#14406f" stroke="#68a0ea" stroke-width="1.5"/>' +
    '<g class="ges-layer" style="animation-delay:1.3s"><path d="M0 26 L34 10 L68 26 L34 42 Z" fill="var(--ges-a)" fill-opacity=".9" stroke="#bff0b4" stroke-width="1.5"/>' +
    '<path d="M17 18 L51 34 M34 10 L34 42 M51 18 L17 34" stroke="#0b1a35" stroke-opacity=".45" stroke-width="1"/></g></g>' +
    '<circle class="ges-ring" cx="150" cy="72" r="16" stroke="var(--ges-a)" stroke-width="2"/>' +
    '<g class="ges-pin"><path d="M152 70 c-9 -10 -14 -16 -14 -24 a14 14 0 0 1 28 0 c0 8 -5 14 -14 24 z" fill="var(--ges-a)" stroke="#eafbe5" stroke-width="1.5"/>' +
    '<circle cx="152" cy="46" r="5" fill="#0b1a35"/></g>' +
    '<defs><linearGradient id="ges-sweep" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--ges-a)" stop-opacity=".0"/>' +
    '<stop offset="1" stop-color="var(--ges-a)" stop-opacity=".22"/></linearGradient></defs></svg>';

  // the name's second part in the product colour: the last word ("GeoEstate CRM", "Land Registry"), or after the suite's
  // prefix in a single word ("PolyGIS", "Geohub", "FarmScan", "LandCheck")
  var name = /\s/.test(product.trim()) ? esc(product).replace(/^(.*\s)(\S+)$/, "$1<b>$2</b>")
    : esc(product).replace(/^(Poly|Geo|Land|Farm)(.+)$/, "$1<b>$2</b>");
  var root = document.createElement("div");
  root.className = "ges" + (warm ? " is-warm" : "") + (still ? " is-still" : "");
  root.style.setProperty("--ges-a", accent);
  root.setAttribute("role", "status");
  root.setAttribute("aria-live", "polite");
  root.setAttribute("aria-busy", "true");
  root.setAttribute("aria-label", "Starting " + product);
  root.innerHTML =
    '<div class="ges-grid"></div>' +
    '<div class="ges-win"><div class="ges-body"><div class="ges-art">' + art + '</div>' +
    '<div class="ges-txt"><div class="ges-brand"><img src="' + esc(ds.logo || "/logo.png") + '" alt=""><span>GeoEstate</span></div>' +
    '<h1 class="ges-name">' + name + '</h1>' +
    (ds.tagline ? '<p class="ges-tag">' + esc(ds.tagline) + "</p>" : "") +
    '<span class="ges-ed">' + esc(ds.edition || "") + "</span></div></div>" +
    '<div class="ges-foot"><div class="ges-status"><span class="ges-msg">Starting ' + esc(product) + '…</span><span><button type="button" class="ges-skip" tabindex="-1">Skip</button> <span class="ges-pct">0 %</span></span></div>' +
    '<div class="ges-bar"><i></i></div>' +
    '<div class="ges-legal"><span>' + (ds.version ? "Version " + esc(ds.version) + " · " : "") + "Part of the GeoEstate suite</span>" +
    "<span>© " + new Date().getFullYear() + " GeoEstate NIG Limited. All rights reserved.</span></div></div></div>";

  var style = document.createElement("style");
  style.textContent = css;
  (document.head || document.documentElement).appendChild(style);
  (document.body || document.documentElement).appendChild(root);
  if (!document.body) document.addEventListener("DOMContentLoaded", function () { document.body.appendChild(root); });

  var bar = root.querySelector(".ges-bar i"), msg = root.querySelector(".ges-msg"), pctEl = root.querySelector(".ges-pct");
  var skip = root.querySelector(".ges-skip");
  var pct = 0, ready = false, closing = false, toldAt = 0;
  function show(p) { pct = Math.max(pct, Math.min(100, p)); bar.style.width = pct + "%"; pctEl.textContent = Math.round(pct) + " %"; }
  // the usual steps of starting, shown in turn through the launch, between what the application says it is doing
  var before = ["Starting " + product, "Loading the interface", "Connecting to GeoEstate ID", "Checking your licence"];
  var after = ["Loading your organization's settings", "Loading map and analysis components", "Preparing your workspace", "Almost ready"];
  var shown = -1, readyAt = 0;
  var tick = setInterval(function () {
    if (closing) return;
    var f = Math.min(1, (Date.now() - t0) / minMs);          // how far through the launch, by time
    var eased = 1 - Math.pow(1 - f, 1.6);                   // a little quicker at first, as starting software is
    show(ready ? 3 + 97 * eased : Math.min(94, 3 + 91 * eased));
    if (Date.now() - toldAt > 1600) {
      // before the application is ready: its first steps; after: the rest, spread over the time left; never an earlier step
      var list = ready ? after : before;
      var from = ready ? readyAt : t0, span = Math.max(1, (ready ? t0 + minMs : t0 + minMs * 0.6) - from);
      var i = Math.min(list.length - 1, Math.floor((Date.now() - from) / span * list.length)) + (ready ? before.length : 0);
      if (i > shown) { shown = i; msg.textContent = (i < before.length ? before[i] : after[i - before.length]) + "…"; }
    }
    if (ready && f >= 1) close();
  }, 100);
  show(3);

  function close() {
    if (closing) return;
    closing = true;
    clearInterval(tick);
    msg.textContent = "Ready";
    show(100);
    setTimeout(function () {
      root.classList.add("is-out");
      root.setAttribute("aria-busy", "false");
      document.removeEventListener("keydown", onKey);
      setTimeout(function () { if (root.parentNode) root.parentNode.removeChild(root); }, 420);
    }, still ? 0 : 350);
  }
  function onKey(e) { if (ready && (e.key === "Escape" || e.key === "Enter" || e.key === " ")) close(); }
  function done() {
    if (ready) return;
    ready = true;
    readyAt = Date.now();
    shown = Math.max(shown, before.length - 1);
    if (Date.now() - t0 >= minMs) { close(); return; }
    skip.classList.add("is-on");                            // the application is ready: the rest of the launch can be skipped
    skip.tabIndex = 0;
    skip.addEventListener("click", close);
    root.addEventListener("click", function (e) { if (!(e.target.closest && e.target.closest(".ges-win"))) close(); });   // outside the window
    document.addEventListener("keydown", onKey);
  }
  window.GeoEstateSplash = {
    step: function (text) { if (closing) return; toldAt = Date.now(); if (text) msg.textContent = text + "…"; },
    edition: function (text) { var e = root.querySelector(".ges-ed"); if (e) e.textContent = text || ""; },
    done: done,
  };
  if (ds.wait === "app") {
    setTimeout(done, minMs + 15000);                      // never in the way if the application forgets to say it is ready
  } else if (document.readyState === "complete") {
    done();
  } else {
    window.addEventListener("load", done);
  }
})();
