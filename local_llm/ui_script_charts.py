"""Front-end: uiChart, the one chart primitive (bar, line, sparkline).

The UI had exactly one chart (the prefill curve, hand-built rows in
ui_script_tasks.py) and Wave 4 adds several more: the admission queue's
history, benchmark throughput per model, dashboards. Each hand-rolled chart
drifted in colour (hex values that ignored dark mode), stroke width and
empty-state wording, so this is the single implementation every panel calls:

    uiChart(host, {type, series: [{label, values}], labels, unit})

  * type     "bar" (horizontal rows: label, track, value; long model names stay
             readable on a phone), "line" (SVG, shared y axis from 0, first /
             last x labels) or "sparkline" (small SVG line + area, no axes,
             latest value beside it). Default "line".
  * series   up to four [{label, values: [number|null]}]. Colour follows the
             series' POSITION (s1..s4 -> --chart-1..4), never its rank, and a
             fifth series is dropped rather than given a generated colour.
             null values are gaps: a line breaks there, a bar shows "-".
  * labels   category names (bar) or x positions (line) in values' order.
  * unit     appended to every printed value ("tok/s", "ms", "%", "GB").
  * optional: max (fixed scale top, e.g. 100 for a percentage), min (line
             only; default 0 or the data minimum when negative), height (px of
             the plot), format(v) -> string, title (accessible name), empty
             (text when there is nothing to draw), last (sparkline: show the
             latest value, default true).

It replaces host's children and returns the chart's root element. Colours,
stroke widths and the track live in ui_styles.py (the `charts` block) as
classes over the --chart-* tokens, so light/dark follow the theme; the only
inline geometry is the --fill custom property of a bar (its length) and SVG
coordinate attributes. Every mark carries a <title> (or the row a title
attribute) so a value is readable on hover without colour, and values are
printed as text, never in the series colour.

uiChartPush(list, value, cap) appends to a bounded history array (the ring
buffer a live sparkline needs) and returns it.

Function declarations hoist across the single assembled <script>; this part has
no top-level statements besides one constant.
"""

from __future__ import annotations

UI_JS_CHARTS = r"""   // ------------------------------------------------------------ charts ---

   var UI_SVG_NS = "http://www.w3.org/2000/svg";

   function uiSvg(tag, attrs) {
     var el = document.createElementNS(UI_SVG_NS, tag);
     if (attrs) {
       Object.keys(attrs).forEach(function(k) { el.setAttribute(k, String(attrs[k])); });
     }
     return el;
   }

   function uiChartNum(v) {
     return typeof v === "number" && isFinite(v);
   }

   // "12.3 tok/s", "87%", "-" for a gap. Large numbers lose their decimals:
   // a chart label is read at a glance, not copied.
   function uiChartFormat(v, spec) {
     if (!uiChartNum(v)) return "-";
     if (spec && typeof spec.format === "function") return String(spec.format(v));
     var a = Math.abs(v);
     var n = a >= 100 ? Math.round(v) : (a >= 10 ? Math.round(v * 10) / 10 : Math.round(v * 100) / 100);
     var unit = spec && spec.unit ? String(spec.unit) : "";
     if (!unit) return String(n);
     return n + (unit === "%" ? "" : " ") + unit;
   }

   function uiChartPush(list, value, cap) {
     list = list || [];
     list.push(uiChartNum(value) ? value : null);
     var max = cap || 60;
     while (list.length > max) list.shift();
     return list;
   }

   function uiChartEl(tag, cls, text) {
     var el = document.createElement(tag);
     if (cls) el.className = cls;
     if (text != null) el.textContent = String(text);
     return el;
   }

   function uiChartLegend(series) {
     var legend = uiChartEl("div", "uichart-legend");
     series.forEach(function(s, i) {
       legend.appendChild(uiChartEl("span", "uichart-key s" + (i + 1), s.label || ("series " + (i + 1))));
     });
     return legend;
   }

   function uiChartTop(series, spec) {
     var top = uiChartNum(spec.max) ? spec.max : 0;
     if (!uiChartNum(spec.max)) {
       series.forEach(function(s) {
         (s.values || []).forEach(function(v) { if (uiChartNum(v) && v > top) top = v; });
       });
     }
     return top > 0 ? top : 1;
   }

   function uiChart(host, spec) {
     if (typeof host === "string") host = document.getElementById(host);
     if (!host) return null;
     spec = spec || {};
     var type = spec.type || "line";
     // Four positions, four tokens; never a generated fifth colour.
     var series = (spec.series || []).slice(0, 4);
     var labels = spec.labels || [];
     var root = uiChartEl("div", "uichart uichart-" + type);
     root.setAttribute("role", "img");
     if (spec.title) root.setAttribute("aria-label", String(spec.title));
     host.textContent = "";
     host.appendChild(root);

     var any = series.some(function(s) {
       return (s.values || []).some(function(v) { return uiChartNum(v); });
     });
     if (!any) {
       root.appendChild(uiChartEl("p", "uichart-empty", spec.empty || "No data yet."));
       return root;
     }
     if (type === "bar") uiChartBars(root, series, labels, spec);
     else if (type === "sparkline") uiChartSpark(root, series, spec);
     else uiChartLines(root, series, labels, spec);
     return root;
   }

   // Horizontal bars: one group per label, one track per series inside it.
   function uiChartBars(root, series, labels, spec) {
     if (series.length > 1) root.appendChild(uiChartLegend(series));
     var top = uiChartTop(series, spec);
     var rows = uiChartEl("div", "uichart-rows");
     var count = 0;
     series.forEach(function(s) { count = Math.max(count, (s.values || []).length); });
     for (var i = 0; i < count; i++) {
       var name = labels[i] != null ? String(labels[i]) : String(i + 1);
       var group = uiChartEl("div", "uichart-group");
       var tip = [];
       series.forEach(function(s, j) {
         var v = (s.values || [])[i];
         tip.push((series.length > 1 ? (s.label || "") + " " : "") + uiChartFormat(v, spec));
         var label = uiChartEl("span", "uichart-label", j === 0 ? name : "");
         var track = uiChartEl("span", "uichart-track s" + (j + 1));
         var fill = uiChartEl("span", "uichart-fill");
         var pct = uiChartNum(v) ? Math.max(0, Math.min(100, (v / top) * 100)) : 0;
         // Chart geometry is the one inline style the kit allows.
         fill.style.setProperty("--fill", pct.toFixed(1) + "%");
         track.appendChild(fill);
         group.appendChild(label);
         group.appendChild(track);
         group.appendChild(uiChartEl("span", "uichart-value mono", uiChartFormat(v, spec)));
       });
       group.setAttribute("title", name + ": " + tip.join(", "));
       rows.appendChild(group);
     }
     root.appendChild(rows);
   }

   // The points of one series in a W x H box; null values split segments.
   function uiChartSegments(values, n, lo, hi, w, h) {
     var segs = [];
     var cur = [];
     var span = hi - lo || 1;
     for (var i = 0; i < values.length; i++) {
       var v = values[i];
       if (!uiChartNum(v)) {
         if (cur.length) segs.push(cur);
         cur = [];
         continue;
       }
       var x = n > 1 ? (i / (n - 1)) * w : w / 2;
       var y = h - ((v - lo) / span) * h;
       cur.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
     }
     if (cur.length) segs.push(cur);
     return segs;
   }

   function uiChartPath(svg, seg, h, area) {
     var pts = seg.map(function(p) { return p[0] + "," + p[1]; }).join(" ");
     if (area && seg.length > 1) {
       var first = seg[0];
       var last = seg[seg.length - 1];
       svg.appendChild(uiSvg("polygon", {
         "class": "uichart-area",
         points: first[0] + "," + h + " " + pts + " " + last[0] + "," + h
       }));
     }
     if (seg.length === 1) {
       // A lone point between gaps: a short tick so it is still visible.
       svg.appendChild(uiSvg("line", {
         "class": "uichart-line", x1: seg[0][0] - 1, y1: seg[0][1], x2: seg[0][0] + 1, y2: seg[0][1]
       }));
       return;
     }
     svg.appendChild(uiSvg("polyline", { "class": "uichart-line", points: pts }));
   }

   function uiChartLines(root, series, labels, spec) {
     var W = 300;
     var H = Math.max(40, Number(spec.height) || 120);
     var n = 0;
     var lo = uiChartNum(spec.min) ? spec.min : 0;
     series.forEach(function(s) {
       n = Math.max(n, (s.values || []).length);
       (s.values || []).forEach(function(v) { if (uiChartNum(v) && v < lo) lo = v; });
     });
     var hi = uiChartTop(series, spec);
     if (series.length > 1) root.appendChild(uiChartLegend(series));
     var plot = uiChartEl("div", "uichart-plot");
     var yaxis = uiChartEl("div", "uichart-yaxis mono");
     yaxis.appendChild(uiChartEl("span", "", uiChartFormat(hi, spec)));
     yaxis.appendChild(uiChartEl("span", "", uiChartFormat(lo, spec)));
     var svg = uiSvg("svg", {
       "class": "uichart-svg", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none",
       width: "100%", height: H, focusable: "false"
     });
     [0, 0.5, 1].forEach(function(f) {
       svg.appendChild(uiSvg("line", { "class": "uichart-grid", x1: 0, x2: W, y1: f * H, y2: f * H }));
     });
     series.forEach(function(s, j) {
       var g = uiSvg("g", { "class": "s" + (j + 1) });
       uiChartSegments(s.values || [], n, lo, hi, W, H).forEach(function(seg) {
         uiChartPath(g, seg, H, series.length === 1);
       });
       svg.appendChild(g);
     });
     // Hover: one hit column per x position (wider than the line), its
     // <title> lists every series at that point.
     var step = n > 1 ? W / (n - 1) : W;
     for (var i = 0; i < n; i++) {
       var x = n > 1 ? (i / (n - 1)) * W : W / 2;
       var hit = uiSvg("rect", {
         "class": "uichart-hit", x: Math.max(0, x - step / 2), y: 0,
         width: Math.min(step, W), height: H
       });
       var parts = series.map(function(s) {
         return (series.length > 1 ? (s.label || "") + " " : "") + uiChartFormat((s.values || [])[i], spec);
       });
       var t = uiSvg("title");
       t.textContent = (labels[i] != null ? labels[i] + ": " : "") + parts.join(", ");
       hit.appendChild(t);
       svg.appendChild(hit);
     }
     plot.appendChild(yaxis);
     plot.appendChild(svg);
     plot.appendChild(uiChartEl("span", ""));
     var xaxis = uiChartEl("div", "uichart-xaxis mono");
     if (labels.length) {
       xaxis.appendChild(uiChartEl("span", "", labels[0]));
       if (labels.length > 2) xaxis.appendChild(uiChartEl("span", "", labels[Math.floor((labels.length - 1) / 2)]));
       if (labels.length > 1) xaxis.appendChild(uiChartEl("span", "", labels[labels.length - 1]));
     }
     plot.appendChild(xaxis);
     root.appendChild(plot);
   }

   function uiChartSpark(root, series, spec) {
     var W = 120;
     var H = Math.max(12, Number(spec.height) || 28);
     var n = 0;
     series.forEach(function(s) { n = Math.max(n, (s.values || []).length); });
     var lo = uiChartNum(spec.min) ? spec.min : 0;
     var hi = uiChartTop(series, spec);
     var svg = uiSvg("svg", {
       "class": "uichart-svg", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none",
       width: "100%", height: H, focusable: "false"
     });
     series.forEach(function(s, j) {
       var g = uiSvg("g", { "class": "s" + (j + 1) });
       uiChartSegments(s.values || [], n, lo, hi, W, H).forEach(function(seg) {
         uiChartPath(g, seg, H, j === 0);
       });
       svg.appendChild(g);
     });
     var first = series[0] || {};
     var vals = first.values || [];
     var latest = null;
     for (var i = vals.length - 1; i >= 0; i--) { if (uiChartNum(vals[i])) { latest = vals[i]; break; } }
     var t = uiSvg("title");
     t.textContent = (first.label ? first.label + ": " : "") + uiChartFormat(latest, spec)
                     + " (peak " + uiChartFormat(hi, spec) + ")";
     svg.appendChild(t);
     root.appendChild(svg);
     if (spec.last !== false) root.appendChild(uiChartEl("span", "uichart-value mono", uiChartFormat(latest, spec)));
   }
"""

__all__ = ["UI_JS_CHARTS"]
