import json
import math

from fastapi.responses import JSONResponse
from nicegui import ui, app, run
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.model import SemanticForagingModel

# ---------------------------------------------------------------------------
# Shared chart store — written by Python, read by /chart-data HTTP route.
# The browser's polling script fetches this; no socket.io / run_javascript.
# ---------------------------------------------------------------------------
_chart_data: dict = {
    'ready':  False,
    'data':   [],
    'layout': {},
    'labels': {},   # last formatted label strings — pre-populate new clients instantly
}


@app.get('/chart-data')
async def chart_data_endpoint():
    return JSONResponse(_chart_data)


# ---------------------------------------------------------------------------
# CSS / JS injected into <head> — must live INSIDE the @ui.page function
# in NiceGUI 3.x (module-level add_head_html conflicts with @ui.page).
# ---------------------------------------------------------------------------
_HEAD_HTML = '''
<style>
  body { background-color:#0f172a; color:#f8fafc; font-family:"Inter",sans-serif; }
  .metric-card {
    background:rgba(30,41,59,0.7); backdrop-filter:blur(10px);
    border:1px solid rgba(255,255,255,0.1); border-radius:12px;
    padding:20px; transition:transform 0.3s ease;
  }
  .metric-card:hover { transform:translateY(-5px); border-color:rgba(139,92,246,0.5); }
  .metric-title { font-size:.875rem; text-transform:uppercase; letter-spacing:.05em; color:#94a3b8; }
  .metric-value { font-size:2rem; font-weight:700; color:#f8fafc; margin-top:5px; }
  .header-text  { font-size:2.5rem; font-weight:800; color:#ffffff; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap"
      rel="stylesheet">
<script>
/* Suppress null/non-Error unhandled rejections (NiceGUI eval re-throws null
   on its first JS eval; capture=true runs BEFORE NiceGUI's bubble listener) */
;(function(){
  window.addEventListener('unhandledrejection', function(evt){
    if (!evt.reason || !(evt.reason instanceof Error)){
      evt.preventDefault();
      evt.stopImmediatePropagation();
    }
  }, true);
  var _oe = window.onerror;
  window.onerror = function(msg, src, line, col, err){
    if (err === null || err === undefined) return true;
    return _oe ? _oe.apply(this, arguments) : false;
  };
})();
</script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script>
(function () {
  function pollChart() {
    fetch('/chart-data')
      .then(function(r){ return r.json(); })
      .then(function(fig){
        if (!fig.ready) return;
        var div = document.getElementById('sfeef-chart');
        if (!div || typeof Plotly === 'undefined') return;
        Plotly.react(div, fig.data, fig.layout, {responsive:true})
          .catch(function(e){ console.warn('Plotly error:', e); });
      })
      .catch(function(){});   /* server not ready yet — ignore */
  }
  document.addEventListener('DOMContentLoaded', function(){
    pollChart();
    setInterval(pollChart, 800);
  });
})();
</script>
'''


class SemanticApp:
    def __init__(self):
        self.metrics = None
        ui.add_head_html(_HEAD_HTML)
        ui.colors(
            primary='#3b82f6', secondary='#8b5cf6', accent='#f59e0b',
            dark='#0f172a', positive='#10b981', negative='#ef4444',
            info='#3b82f6', warning='#f59e0b',
        )
        self.build_ui()
        # If a previous simulation result is cached, populate labels right away
        # so new client connections don't see "—" while waiting for the timer.
        if _chart_data['ready']:
            self._apply_cached_labels()
        # Fire 1 s after the client finishes the WebSocket handshake.
        ui.timer(1.0, self.run_simulation, once=True)

    async def run_simulation(self):
        try:
            ui.notify('Running S-FEEF Predictor… Compressing Information states.', color='info')

            mode_val           = self.mode_select.value
            timescale_val      = self.timescale_slider.value
            fdr_val            = self.fdr_slider.value
            eat_radius_val     = self.eat_radius_slider.value
            dead_entropy_val   = self.dead_entropy_input.value
            agent_fe_val       = self.agent_fe_input.value
            logical_target_val = self.logical_target_select.value
            alpha_val          = self.feef_alpha_slider.value

            def compute_metrics():
                model = SemanticForagingModel(
                    mode=mode_val,
                    timescale=int(timescale_val),
                    food_disappear_rate=fdr_val,
                    eat_radius=int(eat_radius_val),
                    dead_entropy=dead_entropy_val,
                    agentlevel_fe=agent_fe_val,
                )
                initp = model.get_initial_distribution(
                    agentinitloc=2,
                    initial_foodloc='uniform',
                    logical_target=logical_target_val,
                )
                return model.compute_pareto_front(initp, alpha=alpha_val)

            self.metrics = await run.io_bound(compute_metrics)
            self._push_dashboard()
            ui.notify('S-FEEF Optimization Complete!', color='positive')

        except Exception as e:
            ui.notify(f'Simulation error: {str(e)}', color='negative')

    def _apply_cached_labels(self):
        """Pre-populate labels from _chart_data for new client connections."""
        lb = _chart_data.get('labels', {})
        if not lb:
            return
        self.mi_label.set_text(lb.get('mi', '—'))
        self.si_label.set_text(lb.get('si', '—'))
        self.kwsi_label.set_text(lb.get('kwsi', '—'))
        self.feef_label.set_text(lb.get('feef', '—'))
        self.alpha_nat_label.set_text(lb.get('alpha_nat', 'α_c (dark room) = —'))
        self.alpha_crit_label.set_text(lb.get('alpha_crit', '—'))

    @staticmethod
    def _safe_float(v, fallback: float = 0.0) -> float:
        try:
            f = float(v)
            return f if math.isfinite(f) else fallback
        except Exception:
            return fallback

    def _push_dashboard(self):
        """Update NiceGUI labels and write new figure JSON into _chart_data.

        Chart data is served via /chart-data (plain HTTP).  The browser polling
        script picks it up within ~800 ms — zero socket.io / run_javascript
        involvement for the chart, which was the source of every crash so far.
        """
        global _chart_data

        if not self.metrics:
            return

        try:
            self.mi_label.set_text(f"{self.metrics['actual_mi']:.2f} bits")
            self.si_label.set_text(f"{self.metrics['semantic_saturation_threshold']:.2f} bits")
            self.kwsi_label.set_text(f"{self.metrics['kw_semantic_information']:.2f} V-units")
            self.feef_label.set_text(f"{self.metrics['best_feef_score']:.2f}")

            alpha_c = self.metrics.get('alpha_crit', float('inf'))
            ac_str  = f"{alpha_c:.2f}" if math.isfinite(alpha_c) else "∞"
            self.alpha_nat_label.set_text(f"α_c (dark room) = {ac_str}")
            self.alpha_crit_label.set_text(ac_str)

            ks = self.metrics['info_curve_ks']
            vs = self.metrics['info_curve_vs']
            fs = self.metrics['feef_curve_fs']
            if not ks or not vs or not fs:
                return

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=ks, y=vs, mode='lines+markers', name='Viability (Benefit)',
                line=dict(color='#8b5cf6', width=3),
                marker=dict(symbol='circle', size=8, color='#8b5cf6'),
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=ks, y=fs, mode='lines+markers', name='S-FEEF Score (Cost)',
                line=dict(color='#ef4444', width=2, dash='dash'),
                marker=dict(symbol='diamond', size=8, color='#ef4444'),
            ), secondary_y=True)

            sst    = self._safe_float(self.metrics['semantic_saturation_threshold'],
                                      fallback=max(ks) * 0.95)
            i_star = self._safe_float(self.metrics['best_feef_mi'], fallback=0.0)
            if math.isfinite(sst) and sst > 0:
                fig.add_vline(x=sst, line_width=1, line_dash="solid", line_color="#10b981",
                              annotation_text="SST", annotation_position="top left",
                              annotation_font=dict(color="#10b981", size=11))
            if math.isfinite(i_star):
                fig.add_vline(x=i_star, line_width=1, line_dash="solid", line_color="#f59e0b",
                              annotation_text="I* (min S-FEEF)",
                              annotation_position="bottom right",
                              annotation_font=dict(color="#f59e0b", size=11))

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=70, r=64, t=48, b=60),
                title=dict(text="S-FEEF Policy Selection: Cost vs Viability",
                           font=dict(color='#f8fafc', size=14), x=0.5, xanchor='center'),
                hovermode="x unified",
                hoverlabel=dict(bgcolor='#1e293b', bordercolor='#475569',
                                font=dict(color='#f8fafc', size=12)),
                xaxis=dict(title="I(X₀; Y₀|π)  |  Cost (bits)",
                           gridcolor='rgba(255,255,255,0.1)', color='#94a3b8', automargin=True),
                yaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.1)', automargin=True),
                yaxis2=dict(color='#94a3b8', showgrid=False, automargin=True),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                            font=dict(color='#f8fafc', size=11), bgcolor='rgba(15,23,42,0.6)'),
                font=dict(color='#94a3b8'),
            )
            fig.update_yaxes(title_text="V(τ) | Viability", secondary_y=False,
                             gridcolor='rgba(255,255,255,0.1)', automargin=True)
            fig.update_yaxes(title_text="S-FEEF Score", secondary_y=True, showgrid=False,
                             automargin=True)

            # Strip heavy template (~7 KB) before storing.
            fd = fig.to_plotly_json()
            fd.get('layout', {}).pop('template', None)
            _chart_data['data']   = fd.get('data', [])
            _chart_data['layout'] = fd.get('layout', {})
            _chart_data['labels'] = {
                'mi':        f"{self.metrics['actual_mi']:.2f} bits",
                'si':        f"{self.metrics['semantic_saturation_threshold']:.2f} bits",
                'kwsi':      f"{self.metrics['kw_semantic_information']:.2f} V-units",
                'feef':      f"{self.metrics['best_feef_score']:.2f}",
                'alpha_nat': f"α_c (dark room) = {ac_str}",
                'alpha_crit': ac_str,
            }
            _chart_data['ready']  = True

        except Exception as e:
            ui.notify(f'Dashboard error: {str(e)}', color='warning')

    def build_ui(self):
        with ui.header().classes(
            'bg-dark/80 backdrop-blur border-b border-gray-800 p-4 w-full '
            'flex items-center justify-between'
        ):
            ui.label('S-FEEF Framework & Semantic Information').classes('header-text')
            ui.button('Run S-FEEF Optimization', on_click=self.run_simulation,
                      icon='psychology').classes(
                'rounded-full bg-gradient-to-r from-blue-500 to-purple-500 '
                'text-white font-bold px-6 py-2 shadow-lg '
                'hover:shadow-purple-500/50 transition-all'
            )

        with ui.row().classes('w-full max-w-7xl mx-auto mt-8 gap-8 items-start'):

            # ---- Left: parameters ----
            with ui.column().classes('w-full md:w-1/4 gap-4'):
                ui.label('Environment & Physiology').classes(
                    'text-xl font-bold text-slate-200 border-b border-slate-700 pb-2 w-full'
                )
                with ui.card().classes('w-full bg-slate-800/50 backdrop-blur border border-slate-700'):
                    self.mode_select = ui.select(
                        {'pos': 'Positive (Seeks target)', 'neg': 'Negative (Avoids)',
                         'neutral': 'Neutral'},
                        label='Survival Mode', value='pos',
                    ).classes('w-full')
                    self.logical_target_select = ui.select(
                        {'food': 'Aligned to food (Prior)', 'uniform': 'Delusional (Uniform)',
                         '0': 'Always Target 0', '4': 'Always Target 4'},
                        label='Logical Target Prior', value='food',
                    ).classes('w-full mt-2')

                    ui.label('Timescale (τ)').classes('text-sm text-slate-400 mt-4')
                    self.timescale_slider = ui.slider(min=1, max=15, value=5).props('color="primary"')
                    ui.label().bind_text_from(
                        self.timescale_slider, 'value', backward=lambda v: f'{v} steps'
                    ).classes('text-right w-full text-xs text-slate-300')

                    ui.label('Food Disappear Rate').classes('text-sm text-slate-400 mt-2')
                    self.fdr_slider = ui.slider(min=0.0, max=0.5, step=0.01, value=0.1).props('color="secondary"')
                    ui.label().bind_text_from(
                        self.fdr_slider, 'value', backward=lambda v: f'Rate: {v:.2f}'
                    ).classes('text-right w-full text-xs text-slate-300')

                    ui.label('Eat Radius').classes('text-sm text-slate-400 mt-2')
                    self.eat_radius_slider = ui.slider(min=0, max=2, step=1, value=1).props(
                        'color="accent" markers snap'
                    )
                    ui.label().bind_text_from(
                        self.eat_radius_slider, 'value', backward=lambda v: f'Radius: {int(v)}'
                    ).classes('text-right w-full text-xs text-slate-300')

                    ui.label('Physiological Cost Base (Agent Level FE)').classes('text-sm text-slate-400 mt-2')
                    self.agent_fe_input = ui.number(value=10).classes('w-full')

                    ui.label('Dead Entropy (Penalty)').classes('text-sm text-slate-400 mt-2')
                    self.dead_entropy_input = ui.number(value=100.0).classes('w-full')

                    ui.label('S-FEEF Metabolic Multiplier (α)').classes('text-sm text-slate-400 mt-4')
                    self.feef_alpha_slider = ui.slider(min=0.0, max=20.0, step=0.1, value=4.0).props(
                        'color="negative"'
                    )
                    ui.label().bind_text_from(
                        self.feef_alpha_slider, 'value', backward=lambda v: f'Alpha: {v}'
                    ).classes('text-right w-full text-xs text-slate-300')

            # ---- Centre: metrics ----
            with ui.column().classes('w-full md:w-1/4 gap-4'):
                ui.label('Formal Metrics').classes(
                    'text-xl font-bold text-slate-200 border-b border-slate-700 pb-2 w-full'
                )
                with ui.column().classes('w-full gap-4'):
                    with ui.card().classes('metric-card w-full'):
                        ui.label('Mutual Information (Shannon)').classes('metric-title')
                        self.mi_label = ui.label('—').classes('metric-value')
                    with ui.card().classes('metric-card w-full'):
                        ui.label('Semantic Saturation Threshold').classes('metric-title text-emerald-400')
                        self.si_label = ui.label('—').classes('metric-value text-emerald-400')
                        ui.label('K&W SI (V-units)').classes('text-xs text-emerald-300 mt-1')
                        self.kwsi_label = ui.label('—').classes('text-sm font-bold text-emerald-300')
                    with ui.card().classes('metric-card w-full'):
                        ui.label('Optimal S-FEEF Cost').classes('metric-title text-rose-400')
                        self.feef_label = ui.label('—').classes('metric-value text-rose-400')
                        self.alpha_nat_label = ui.label('α_c (dark room) = —').classes(
                            'text-xs text-rose-300 font-bold mt-1'
                        )
                    with ui.card().classes('metric-card w-full'):
                        ui.label('Critical Alpha (α_c)').classes('metric-title text-amber-400')
                        self.alpha_crit_label = ui.label('—').classes('metric-value text-amber-400')
                        ui.label('Dark room threshold').classes('text-xs text-amber-300 mt-1')

            # ---- Right: chart ----
            with ui.column().classes('w-full md:w-5/12 gap-4 flex-grow'):
                ui.label('S-FEEF Thermodynamic Minimization').classes(
                    'text-xl font-bold text-slate-200 border-b border-slate-700 pb-2 w-full'
                )
                with ui.card().classes(
                    'w-full bg-slate-800/80 backdrop-blur border border-slate-700 p-0'
                ):
                    # Plotly CDN renders here via the polling script in _HEAD_HTML.
                    # No nicegui-plotly bundle, no run_javascript.
                    ui.html(
                        '<div id="sfeef-chart" style="width:100%;height:480px;">'
                        '<p style="color:#94a3b8;padding:2rem;text-align:center">'
                        'Running S-FEEF simulation…</p></div>'
                    ).classes('w-full')


@ui.page('/')
def index():
    SemanticApp()


ui.run(
    title='Semantic Info & S-FEEF',
    dark=True,
    port=5000,
    show=False,
    reload=False,
    reconnect_timeout=10.0,
)
