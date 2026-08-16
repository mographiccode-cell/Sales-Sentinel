(() => {
  'use strict';
  const $ = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
  const number = (v) => Number(v ?? 0) || 0;
  const escapeHtml = (v) => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const money = (v) => `${number(v).toLocaleString(undefined,{maximumFractionDigits:2})} SAR`;
  const pct = (v, signed=false) => `${signed && number(v)>0?'+':''}${number(v).toFixed(1)}%`;

  // Mobile navigation
  $$('[data-sidebar-open]').forEach(b => b.addEventListener('click', () => document.body.classList.add('sidebar-open')));
  $$('[data-sidebar-close]').forEach(b => b.addEventListener('click', () => document.body.classList.remove('sidebar-open')));

  // Generic progressive disclosure
  $$('[data-toggle-target]').forEach(button => button.addEventListener('click', () => {
    const panel = $(button.dataset.toggleTarget);
    if (!panel) return;
    panel.hidden = !panel.hidden;
    button.setAttribute('aria-expanded', String(!panel.hidden));
  }));

  // File chooser feedback
  $$('.ux3-file-drop input[type=file]').forEach(input => input.addEventListener('change', () => {
    const label = input.closest('.ux3-file-drop');
    const strong = $('strong', label);
    if (input.files?.[0]) strong.textContent = input.files[0].name;
  }));

  // Analysis workspace tabs
  $$('[data-ux-tab]').forEach(button => button.addEventListener('click', () => {
    const scope = button.closest('.ux3-result-workspace');
    if (!scope) return;
    $$('[data-ux-tab]', scope).forEach(b => b.classList.toggle('active', b === button));
    $$('[data-ux-panel]', scope).forEach(panel => {
      const active = panel.dataset.uxPanel === button.dataset.uxTab;
      panel.classList.toggle('active', active);
      panel.hidden = !active;
    });
  }));

  // Make analysis history rows one-click targets.
  $$('[data-row-href]').forEach(row => {
    row.setAttribute('tabindex','0');
    const open = () => { window.location.href = row.dataset.rowHref; };
    row.addEventListener('click', open);
    row.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });

  // Report date filters (client-side; no page reload).
  const today = new Date();
  const startOfWeek = new Date(today); startOfWeek.setDate(today.getDate() - ((today.getDay()+6)%7)); startOfWeek.setHours(0,0,0,0);
  const startOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);
  $$('[data-report-filter]').forEach(button => button.addEventListener('click', () => {
    $$('[data-report-filter]').forEach(b => b.classList.toggle('active', b === button));
    const mode = button.dataset.reportFilter;
    $$('[data-report-row]').forEach(row => {
      const d = new Date(`${row.dataset.reportDate}T00:00:00`);
      let show = true;
      if (mode === 'today') show = d.toDateString() === today.toDateString();
      if (mode === 'week') show = d >= startOfWeek;
      if (mode === 'month') show = d >= startOfMonth;
      row.hidden = !show;
    });
  }));

  // Persist the most recent instant-analysis result in the browser.
  const cacheKey = 'salesSentinel.latestAnalysis.v3';
  const snapshotNode = $('#instant-analysis-json');
  let latestAnalysis = null;
  if (snapshotNode) {
    try { latestAnalysis = JSON.parse(snapshotNode.textContent); localStorage.setItem(cacheKey, JSON.stringify(latestAnalysis)); } catch (_) {}
  }
  if (!latestAnalysis) { try { latestAnalysis = JSON.parse(localStorage.getItem(cacheKey) || 'null'); } catch (_) {} }

  // Compact cached result only; never duplicate the full analysis workspace.
  if (latestAnalysis) {
    $$('[data-browser-analysis-slot]').forEach(slot => {
      const ar = slot.dataset.locale === 'ar';
      const severity = latestAnalysis.severity || 'low';
      slot.innerHTML = `<section class="ux3-cache-strip severity-${escapeHtml(severity)}">
        <div><span class="ux3-status-dot"></span><strong>${ar?'آخر تحليل محفوظ':'Latest saved analysis'}</strong><small>${escapeHtml(latestAnalysis.source_filename || '')}</small></div>
        <div class="ux3-cache-metrics"><span><small>${ar?'الخطر':'Risk'}</small><b>${pct(latestAnalysis.risk_probability_pct)}</b></span><span><small>${ar?'الانخفاض المتوقع':'Forecast decline'}</small><b>${pct(latestAnalysis.predicted_decline_pct)}</b></span><span><small>${ar?'الجودة':'Quality'}</small><b>${pct(latestAnalysis.forecast_accuracy_pct)}</b></span></div>
      </section>`;
    });
  }

  const makeReportCsv = (a) => {
    const q = (v) => `"${String(v ?? '').replaceAll('"','""')}"`;
    const lines = [['metric','value'].map(q).join(',')];
    [
      ['source_file',a.source_filename],['data_start',a.data_start],['data_end',a.data_end],['history_days',a.history_days],
      ['decline_probability_pct',number(a.risk_probability_pct).toFixed(4)],['observed_decline_pct',number(a.observed_decline_pct).toFixed(4)],
      ['predicted_decline_pct',number(a.predicted_decline_pct).toFixed(4)],['forecast_accuracy_1_minus_wape_pct',number(a.forecast_accuracy_pct).toFixed(4)],
      ['forecast_error_wape_pct',number(a.forecast_error_pct).toFixed(4)],['interval_coverage_pct',number(a.interval_coverage_pct).toFixed(4)],
      ['model_name',a.model_name],['model_version',a.model_version],['alert',a.alert],['severity',a.severity]
    ].forEach(r => lines.push(r.map(q).join(',')));
    lines.push('', ['date','predicted_sales','lower_bound','upper_bound','decline_probability_pct'].map(q).join(','));
    (a.forecasts || []).forEach(i => lines.push([i.date,i.predicted,i.lower,i.upper,i.decline_probability_pct].map(q).join(',')));
    return '\ufeff'+lines.join('\r\n');
  };
  document.addEventListener('click', (e) => {
    const download = e.target.closest('[data-download-analysis]');
    if (download && latestAnalysis) {
      e.preventDefault();
      const blob = new Blob([makeReportCsv(latestAnalysis)], {type:'text/csv;charset=utf-8'});
      const url = URL.createObjectURL(blob); const a=document.createElement('a');
      a.href=url; a.download=`sales-sentinel-${String(latestAnalysis.data_end || 'analysis').replace(/[^\w-]/g,'-')}.csv`;
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }
    if (e.target.closest('[data-print-analysis]') || e.target.closest('[data-print-page]')) { e.preventDefault(); window.print(); }
  });

  // Existing lightweight sales chart.
  const chart = $('.chart[data-sales]');
  if (chart) {
    let sales=[], forecasts=[];
    try { sales = JSON.parse(chart.dataset.sales || '[]'); } catch (_) {}
    try { forecasts = JSON.parse(chart.dataset.forecasts || '[]'); } catch (_) {}
    const points = [...sales.map(x=>({...x,kind:'actual'})), ...forecasts.map(x=>({date:x.date,value:x.median,kind:'forecast'}))];
    const max = Math.max(...points.map(x=>number(x.value)),1);
    chart.innerHTML = points.map(x => `<div class="chart-bar ${x.kind==='forecast'?'forecast':''}" style="height:${Math.max(4,number(x.value)/max*100)}%" title="${escapeHtml(x.date)} · ${money(x.value)}"></div>`).join('');
  }
})();
