import numpy as np
import json


def build_professional_map(
    all_data:      np.ndarray,   # (total_days, 3, H, W) normalized
    day_idx:       int,
    pred_rf:       np.ndarray,   # (H, W) normalized
    lats:          np.ndarray,
    lons:          np.ndarray,
    norm_stats:    dict,
    selected_date: str,
    rf_delta:      int   = 0,
    tmp_delta:     float = 0.0,
    window:        int   = 15    # days before & after to preload
) -> str:

    def denorm(d, s):
        if s["method"] == "minmax":
            return d * (s["max"] - s["min"]) + s["min"]
        return d * s["std"] + s["mean"]

    # Preload window of days for timeline
    total_days = all_data.shape[0]
    start_idx  = max(7, day_idx - window)
    end_idx    = min(total_days - 1, day_idx + window)

    days_rf, days_mt, days_mn, days_dates = [], [], [], []

    for i in range(start_idx, end_idx + 1):
        rf = denorm(all_data[i, 0], norm_stats["rainfall"])
        mt = denorm(all_data[i, 1], norm_stats["max_temp"])
        mn = denorm(all_data[i, 2], norm_stats["min_temp"])

        if rf_delta != 0:
            rf = rf * (1 + rf_delta / 100)
        if tmp_delta != 0:
            mt = mt + tmp_delta
            mn = mn + tmp_delta

        from datetime import date, timedelta
        base    = date(2018, 1, 1)
        d_date  = (base + timedelta(days=i)).strftime("%Y-%m-%d")

        days_rf.append(np.nan_to_num(rf, nan=0.0).tolist())
        days_mt.append(np.nan_to_num(mt, nan=0.0).tolist())
        days_mn.append(np.nan_to_num(mn, nan=0.0).tolist())
        days_dates.append(d_date)

    # Current day index within our window
    current_window_idx = day_idx - start_idx

    # Prediction
    prf = denorm(pred_rf, norm_stats["rainfall"])
    if rf_delta != 0:
        prf = prf * (1 + rf_delta / 100)
    prf = np.nan_to_num(prf, nan=0.0).tolist()

    lats_list = lats.tolist()
    lons_list = lons.tolist()

    # MT min/max for color scaling
    mt_min = float(np.nanmin(denorm(all_data[:, 1], norm_stats["max_temp"])))
    mt_max = float(np.nanmax(denorm(all_data[:, 1], norm_stats["max_temp"])))
    mn_min = float(np.nanmin(denorm(all_data[:, 2], norm_stats["min_temp"])))
    mn_max = float(np.nanmax(denorm(all_data[:, 2], norm_stats["min_temp"])))

    scenario_txt = ""
    if rf_delta != 0:
        scenario_txt += f"Rain {'+' if rf_delta>0 else ''}{rf_delta}% "
    if tmp_delta != 0:
        scenario_txt += f"Temp {'+' if tmp_delta>0 else ''}{tmp_delta}°C"

    html = f"""<!DOCTYPE html>
<!-- uid:{selected_date}-{rf_delta}-{tmp_delta} -->
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:#f0f4f8; }}
  #map {{ width:100%; height:680px; border-radius:12px; }}

  .legend {{
    position:absolute; bottom:80px; left:10px; z-index:1000;
    background:rgba(255,255,255,0.96); border-radius:10px;
    padding:12px 16px; box-shadow:0 2px 8px rgba(0,0,0,0.15);
    min-width:160px; font-size:12px; border:1px solid #e2e8f0;
  }}
  .legend-title {{
    font-weight:700; font-size:11px; color:#1e293b;
    text-transform:uppercase; letter-spacing:0.05em;
    margin-bottom:8px; padding-bottom:6px;
    border-bottom:1px solid #e2e8f0;
  }}
  .legend-item {{
    display:flex; align-items:center; gap:8px;
    margin-bottom:5px; color:#334155; font-size:11px;
  }}
  .legend-color {{
    width:24px; height:12px; border-radius:3px;
    border:1px solid rgba(0,0,0,0.1); flex-shrink:0;
  }}

  .layer-control {{
    position:absolute; top:10px; right:10px; z-index:1000;
    background:rgba(255,255,255,0.97); border-radius:10px;
    padding:12px 16px; box-shadow:0 2px 8px rgba(0,0,0,0.15);
    font-size:12px; min-width:160px; border:1px solid #e2e8f0;
  }}
  .layer-title {{
    font-weight:700; font-size:11px; color:#1e293b;
    text-transform:uppercase; letter-spacing:0.05em;
    margin-bottom:8px; padding-bottom:6px;
    border-bottom:1px solid #e2e8f0;
  }}
  .layer-item {{
    display:flex; align-items:center; gap:8px;
    margin-bottom:6px; cursor:pointer; color:#334155;
  }}
  .layer-item input {{ cursor:pointer; accent-color:#1A73E8; }}
  .layer-item label {{ cursor:pointer; font-size:12px; }}

  .info-panel {{
    position:absolute; top:10px; left:10px; z-index:1000;
    background:rgba(255,255,255,0.97); border-radius:10px;
    padding:12px 16px; box-shadow:0 2px 8px rgba(0,0,0,0.15);
    font-size:12px; min-width:190px; border:1px solid #e2e8f0;
    display:none;
  }}
  .info-title {{
    font-weight:700; font-size:11px; color:#1A73E8;
    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;
  }}
  .info-row {{
    display:flex; justify-content:space-between;
    margin-bottom:4px; color:#334155;
  }}
  .info-key {{ color:#64748b; font-size:11px; }}
  .info-val {{ font-weight:600; font-size:11px; color:#1e293b; }}

  .timeline {{
    position:absolute; bottom:10px; left:50%;
    transform:translateX(-50%); z-index:1000;
    background:rgba(255,255,255,0.97); border-radius:10px;
    padding:10px 20px; box-shadow:0 2px 8px rgba(0,0,0,0.15);
    font-size:12px; min-width:380px; border:1px solid #e2e8f0;
    text-align:center;
  }}
  .timeline-top {{
    display:flex; justify-content:space-between;
    align-items:center; margin-bottom:6px;
  }}
  .timeline-date {{
    font-weight:700; font-size:13px; color:#1A73E8;
  }}
  .timeline-controls {{
    display:flex; gap:8px; align-items:center;
  }}
  .tl-btn {{
    background:#1A73E8; color:white; border:none;
    border-radius:6px; padding:3px 10px; cursor:pointer;
    font-size:12px; font-weight:600;
  }}
  .tl-btn:hover {{ background:#1557b0; }}
  .tl-btn.stop {{ background:#ef4444; }}
  .timeline input[type=range] {{
    width:100%; cursor:pointer; accent-color:#1A73E8;
  }}
  .timeline-labels {{
    display:flex; justify-content:space-between;
    font-size:10px; color:#94a3b8; margin-top:2px;
  }}

  .date-badge {{
    position:absolute; top:10px; left:50%;
    transform:translateX(-50%); z-index:1000;
    background:rgba(26,115,232,0.92); color:white;
    border-radius:20px; padding:5px 16px;
    font-size:12px; font-weight:600;
    box-shadow:0 2px 6px rgba(0,0,0,0.2); white-space:nowrap;
  }}

  .scenario-badge {{
    position:absolute; bottom:85px; right:10px; z-index:1000;
    background:rgba(249,115,22,0.92); color:white;
    border-radius:8px; padding:5px 12px;
    font-size:11px; font-weight:600;
    box-shadow:0 2px 6px rgba(0,0,0,0.2);
    {'display:block' if scenario_txt else 'display:none'};
  }}
</style>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script
  src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
</head>
<body>
<div style="position:relative; border-radius:12px; overflow:hidden;">
  <div id="map"></div>

  <div class="date-badge" id="dateBadge">📅 {selected_date}</div>

  <div class="scenario-badge">{('⚡ What-If: ' + scenario_txt) if scenario_txt else ''}</div>

  <!-- Info panel -->
  <div class="info-panel" id="infoPanel">
    <div class="info-title">📍 Location Info</div>
    <div class="info-row"><span class="info-key">Latitude</span><span class="info-val" id="iLat">—</span></div>
    <div class="info-row"><span class="info-key">Longitude</span><span class="info-val" id="iLon">—</span></div>
    <div class="info-row"><span class="info-key">Rainfall</span><span class="info-val" id="iRF">—</span></div>
    <div class="info-row"><span class="info-key">Max Temp</span><span class="info-val" id="iMT">—</span></div>
    <div class="info-row"><span class="info-key">Min Temp</span><span class="info-val" id="iMN">—</span></div>
    <div class="info-row"><span class="info-key">AI Pred. Rain</span><span class="info-val" id="iPR">—</span></div>
    <div class="info-row"><span class="info-key">AI Confidence</span><span class="info-val" id="iAI">—</span></div>
  </div>

  <!-- Layer control -->
  <div class="layer-control">
    <div class="layer-title">🗂 Layers</div>
    <div class="layer-item">
      <input type="checkbox" id="lRF" checked onchange="toggleLayer('rf')"/>
      <label for="lRF">🌧 Rainfall</label>
    </div>
    <div class="layer-item">
      <input type="checkbox" id="lMT" onchange="toggleLayer('mt')"/>
      <label for="lMT">🌡 Max Temp</label>
    </div>
    <div class="layer-item">
      <input type="checkbox" id="lMN" onchange="toggleLayer('mn')"/>
      <label for="lMN">❄ Min Temp</label>
    </div>
    <div class="layer-item">
      <input type="checkbox" id="lPR" onchange="toggleLayer('pr')"/>
      <label for="lPR">🤖 AI Prediction</label>
    </div>
  </div>

  <!-- Legend -->
  <div class="legend" id="mapLegend">
    <div class="legend-title">🌧 Rainfall (mm/day)</div>
    <div class="legend-item"><div class="legend-color" style="background:#dbeafe"></div>0 – 5</div>
    <div class="legend-item"><div class="legend-color" style="background:#93c5fd"></div>5 – 15</div>
    <div class="legend-item"><div class="legend-color" style="background:#3b82f6"></div>15 – 30</div>
    <div class="legend-item"><div class="legend-color" style="background:#1d4ed8"></div>30 – 60</div>
    <div class="legend-item"><div class="legend-color" style="background:#581c87"></div>&gt; 60</div>
  </div>

  <!-- Timeline -->
  <div class="timeline">
    <div class="timeline-top">
      <span class="timeline-date" id="tlDate">📅 {selected_date}</span>
      <div class="timeline-controls">
        <button class="tl-btn" onclick="stepBack()">◀</button>
        <button class="tl-btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
        <button class="tl-btn stop" onclick="stopAnim()">■ Stop</button>
        <button class="tl-btn" onclick="stepForward()">▶</button>
      </div>
    </div>
    <input type="range" id="tlSlider"
      min="0" max="{len(days_dates)-1}" value="{current_window_idx}"
      oninput="goToDay(parseInt(this.value))"/>
    <div class="timeline-labels">
      <span>{days_dates[0]}</span>
      <span>{days_dates[len(days_dates)//2]}</span>
      <span>{days_dates[-1]}</span>
    </div>
  </div>
</div>

<script>
// ── All preloaded data ──────────────────────────────────────────────────────
const DAYS_RF    = {json.dumps(days_rf)};
const DAYS_MT    = {json.dumps(days_mt)};
const DAYS_MN    = {json.dumps(days_mn)};
const DAYS_DATES = {json.dumps(days_dates)};
const PRF_DATA   = {json.dumps(prf)};
const LATS       = {json.dumps(lats_list)};
const LONS       = {json.dumps(lons_list)};
const MT_MIN     = {mt_min};
const MT_MAX     = {mt_max};
const MN_MIN     = {mn_min};
const MN_MAX     = {mn_max};
let   currentDay = {current_window_idx};
let   animTimer  = null;
let   isPlaying  = false;

// ── Color helpers ───────────────────────────────────────────────────────────
function rainfallColor(v) {{
  if (v <= 0.5) return null;
  if (v < 5)    return [219,234,254,0.45];
  if (v < 15)   return [147,197,253,0.60];
  if (v < 30)   return [59, 130,246,0.70];
  if (v < 60)   return [29, 78, 216,0.78];
  return              [88, 28, 135,0.88];
}}
function tempColor(v, mn, mx) {{
  const t = Math.max(0, Math.min(1, (v-mn)/(mx-mn+0.01)));
  return [Math.round(54+t*200), Math.round(162-t*130), Math.round(235-t*200), 0.65];
}}
function toRgba(c) {{
  return c ? `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{c[3]}})` : 'rgba(0,0,0,0)';
}}

// ── Map init ────────────────────────────────────────────────────────────────
const map = L.map('map', {{ center:[19.0,76.5], zoom:6, zoomControl:true }});
L.tileLayer(
  'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{ attribution:'© CartoDB', maxZoom:18 }}
).addTo(map);

// Maharashtra boundary
L.polyline([
  [15.6,72.6],[15.6,80.4],[22.4,80.4],[22.4,72.6],[15.6,72.6]
], {{ color:'#1A73E8', weight:2.5, dashArray:'6,4', opacity:0.9 }}).addTo(map);

// Cities
[
  {{n:"Mumbai",     la:19.076,lo:72.877}},
  {{n:"Pune",       la:18.520,lo:73.856}},
  {{n:"Nagpur",     la:21.145,lo:79.088}},
  {{n:"Nashik",     la:19.997,lo:73.791}},
  {{n:"Aurangabad", la:19.877,lo:75.343}},
  {{n:"Solapur",    la:17.687,lo:75.906}},
  {{n:"Kolhapur",   la:16.705,lo:74.243}},
  {{n:"Amravati",   la:20.932,lo:77.757}},
].forEach(c => {{
  L.circleMarker([c.la,c.lo], {{
    radius:5, fillColor:'#1A73E8', color:'white',
    weight:2, fillOpacity:0.9, zIndexOffset:1000
  }}).bindTooltip(c.n, {{
    permanent:true, direction:'top',
    className:'city-label', offset:[0,-8]
  }}).addTo(map);
}});

// ── Grid layer groups ───────────────────────────────────────────────────────
const dlat = LATS.length>1 ? Math.abs(LATS[1]-LATS[0]) : 0.25;
const dlon  = LONS.length>1 ? Math.abs(LONS[1]-LONS[0]) : 0.25;

const layerGroups = {{
  rf: L.layerGroup().addTo(map),
  mt: L.layerGroup(),
  mn: L.layerGroup(),
  pr: L.layerGroup()
}};
const activeFlags = {{ rf:true, mt:false, mn:false, pr:false }};

function renderGridTo(data, colorFn, group) {{
  group.clearLayers();
  for (let i=0; i<LATS.length; i++) {{
    for (let j=0; j<LONS.length; j++) {{
      const v = data[i][j];
      const c = colorFn(v);
      if (!c) continue;
      L.rectangle([
        [LATS[i]-dlat/2, LONS[j]-dlon/2],
        [LATS[i]+dlat/2, LONS[j]+dlon/2]
      ], {{
        fillColor:toRgba(c), fillOpacity:c[3],
        stroke:false, interactive:false
      }}).addTo(group);
    }}
  }}
}}

// ── Go to specific day ──────────────────────────────────────────────────────
function goToDay(idx) {{
  currentDay = idx;
  document.getElementById('tlSlider').value = idx;
  const dateStr = DAYS_DATES[idx];
  document.getElementById('tlDate').textContent   = '📅 ' + dateStr;
  document.getElementById('dateBadge').textContent = '📅 ' + dateStr;

  if (activeFlags.rf) renderGridTo(DAYS_RF[idx], rainfallColor, layerGroups.rf);
  if (activeFlags.mt) renderGridTo(DAYS_MT[idx], v=>tempColor(v,MT_MIN,MT_MAX), layerGroups.mt);
  if (activeFlags.mn) renderGridTo(DAYS_MN[idx], v=>tempColor(v,MN_MIN,MN_MAX), layerGroups.mn);
  if (activeFlags.pr) renderGridTo(PRF_DATA,      rainfallColor, layerGroups.pr);
}}

// Initial render
goToDay(currentDay);

// ── Playback controls ───────────────────────────────────────────────────────
function togglePlay() {{
  if (isPlaying) {{ stopAnim(); return; }}
  isPlaying = true;
  document.getElementById('playBtn').textContent = '⏸ Pause';
  animTimer = setInterval(() => {{
    const next = (currentDay + 1) % DAYS_DATES.length;
    goToDay(next);
  }}, 600);
}}

function stopAnim() {{
  isPlaying = false;
  clearInterval(animTimer);
  document.getElementById('playBtn').textContent = '▶ Play';
}}

function stepBack() {{
  stopAnim();
  goToDay(Math.max(0, currentDay-1));
}}

function stepForward() {{
  stopAnim();
  goToDay(Math.min(DAYS_DATES.length-1, currentDay+1));
}}

// ── Layer toggle ────────────────────────────────────────────────────────────
const legends = {{
  rf:`<div class="legend-title">🌧 Rainfall (mm/day)</div>
    <div class="legend-item"><div class="legend-color" style="background:#dbeafe"></div>0–5</div>
    <div class="legend-item"><div class="legend-color" style="background:#93c5fd"></div>5–15</div>
    <div class="legend-item"><div class="legend-color" style="background:#3b82f6"></div>15–30</div>
    <div class="legend-item"><div class="legend-color" style="background:#1d4ed8"></div>30–60</div>
    <div class="legend-item"><div class="legend-color" style="background:#581c87"></div>&gt;60</div>`,
  mt:`<div class="legend-title">🌡 Max Temp (°C)</div>
    <div class="legend-item"><div class="legend-color" style="background:#bfdbfe"></div>Low (&lt;25°C)</div>
    <div class="legend-item"><div class="legend-color" style="background:#fb923c"></div>Mid (25–38°C)</div>
    <div class="legend-item"><div class="legend-color" style="background:#dc2626"></div>High (&gt;38°C)</div>`,
  mn:`<div class="legend-title">❄ Min Temp (°C)</div>
    <div class="legend-item"><div class="legend-color" style="background:#bfdbfe"></div>Low (&lt;15°C)</div>
    <div class="legend-item"><div class="legend-color" style="background:#a78bfa"></div>Mid (15–25°C)</div>
    <div class="legend-item"><div class="legend-color" style="background:#7c3aed"></div>High (&gt;25°C)</div>`,
  pr:`<div class="legend-title">🤖 AI Pred. Rain (mm/day)</div>
    <div class="legend-item"><div class="legend-color" style="background:#dbeafe"></div>0–5</div>
    <div class="legend-item"><div class="legend-color" style="background:#3b82f6"></div>15–30</div>
    <div class="legend-item"><div class="legend-color" style="background:#581c87"></div>&gt;60</div>`
}};

function toggleLayer(key) {{
  const cbIds = {{rf:'lRF', mt:'lMT', mn:'lMN', pr:'lPR'}};
  const cb = document.getElementById(cbIds[key]);
  activeFlags[key] = cb.checked;
  if (cb.checked) {{
    map.addLayer(layerGroups[key]);
    renderGridTo(
      key==='rf' ? DAYS_RF[currentDay] :
      key==='mt' ? DAYS_MT[currentDay] :
      key==='mn' ? DAYS_MN[currentDay] : PRF_DATA,
      key==='mt' ? v=>tempColor(v,MT_MIN,MT_MAX) :
      key==='mn' ? v=>tempColor(v,MN_MIN,MN_MAX) : rainfallColor,
      layerGroups[key]
    );
    document.getElementById('mapLegend').innerHTML = legends[key];
  }} else {{
    map.removeLayer(layerGroups[key]);
  }}
}}

// ── Hover interaction ───────────────────────────────────────────────────────
function nearest(lat, lon, data) {{
  let bi=0,bj=0,bd=1e9;
  for(let i=0;i<LATS.length;i++)
    for(let j=0;j<LONS.length;j++) {{
      const d=Math.abs(LATS[i]-lat)+Math.abs(LONS[j]-lon);
      if(d<bd){{bd=d;bi=i;bj=j;}}
    }}
  return data[bi][bj];
}}

map.on('mousemove', e => {{
  const {{lat,lng:lon}} = e.latlng;
  if(lat<15.5||lat>22.5||lon<72.5||lon>80.5){{
    document.getElementById('infoPanel').style.display='none';
    return;
  }}
  const rf  = nearest(lat,lon,DAYS_RF[currentDay]);
  const mt  = nearest(lat,lon,DAYS_MT[currentDay]);
  const mn  = nearest(lat,lon,DAYS_MN[currentDay]);
  const prf = nearest(lat,lon,PRF_DATA);
  const conf = Math.max(0,100-Math.abs(rf-prf)*2).toFixed(0)+'%';

  document.getElementById('infoPanel').style.display='block';
  document.getElementById('iLat').textContent = lat.toFixed(3)+'°N';
  document.getElementById('iLon').textContent = lon.toFixed(3)+'°E';
  document.getElementById('iRF').textContent  = rf.toFixed(1)+' mm/day';
  document.getElementById('iMT').textContent  = mt.toFixed(1)+'°C';
  document.getElementById('iMN').textContent  = mn.toFixed(1)+'°C';
  document.getElementById('iPR').textContent  = prf.toFixed(1)+' mm/day';
  document.getElementById('iAI').textContent  = conf;
}});

map.on('mouseout', () => {{
  document.getElementById('infoPanel').style.display='none';
}});
</script>

<style>
.city-label {{
  background:rgba(255,255,255,0.92); border:none;
  border-radius:4px; padding:1px 5px;
  font-size:10px; font-weight:600; color:#1e293b;
  box-shadow:0 1px 3px rgba(0,0,0,0.15); white-space:nowrap;
}}
.leaflet-tooltip.city-label::before {{ display:none; }}
</style>
</body>
</html>"""
    return html