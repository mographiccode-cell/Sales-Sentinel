(() => {
  const STORAGE_KEY = 'salesSentinel.latestAnalysis.v1';

  const parseJson = (value, fallback = null) => {
    try { return JSON.parse(value); } catch (_) { return fallback; }
  };

  const escapeHtml = (value) => String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');

  const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const pct = (value, signed = false) => `${signed && number(value) >= 0 ? '+' : ''}${number(value).toFixed(1)}%`;
  const money = (value) => number(value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

  const serverAnalysisNode = document.getElementById('instant-analysis-json');
  let latestAnalysis = null;
  if (serverAnalysisNode && serverAnalysisNode.textContent.trim()) {
    latestAnalysis = parseJson(serverAnalysisNode.textContent.trim());
    if (latestAnalysis && latestAnalysis.available) {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(latestAnalysis)); } catch (_) {}
    }
  }
  if (!latestAnalysis) {
    try { latestAnalysis = parseJson(localStorage.getItem(STORAGE_KEY) || ''); } catch (_) {}
  }

  const renderBrowserSnapshot = (slot, analysis) => {
    if (!analysis || !analysis.available || slot.hidden) return;
    const ar = slot.dataset.locale === 'ar';
    const forecasts = Array.isArray(analysis.forecasts) ? analysis.forecasts : [];
    const alertClass = analysis.alert ? `severity-${escapeHtml(analysis.severity || 'medium')}` : 'safe';
    const alertTitle = analysis.alert
      ? (ar ? 'آخر تنبيه محفوظ في هذا المتصفح' : 'Latest alert saved in this browser')
      : (ar ? 'آخر تحليل محفوظ في هذا المتصفح' : 'Latest analysis saved in this browser');
    const recommendation = ar ? analysis.recommendation_ar : analysis.recommendation_en;
    const rows = forecasts.map(item => `
      <tr>
        <td>${escapeHtml(item.date)}</td>
        <td>${money(item.predicted)}</td>
        <td>${money(item.lower)}</td>
        <td>${money(item.upper)}</td>
        <td>${pct(item.decline_probability_pct)}</td>
      </tr>`).join('');

    slot.innerHTML = `
      <section class="analysis-section browser-analysis">
        <div class="analysis-title-row">
          <div>
            <span class="eyebrow">BROWSER ANALYSIS CACHE</span>
            <h2>${ar ? 'نتيجة آخر ملف تم تحليله' : 'Latest uploaded-file analysis'}</h2>
            <p>${escapeHtml(analysis.source_filename || '')} · ${escapeHtml(analysis.data_start || '')} → ${escapeHtml(analysis.data_end || '')}</p>
          </div>
          <button type="button" class="btn" data-download-analysis>${ar ? 'تنزيل التقرير CSV' : 'Download report CSV'}</button>
        </div>
        <div class="analysis-alert ${alertClass}">
          <strong>${alertTitle}</strong>
          <span>${ar ? 'احتمال الانخفاض' : 'Decline probability'} ${pct(analysis.risk_probability_pct)}</span>
        </div>
        <div class="kpi-grid six">
          <article class="kpi-card risk"><small>${ar ? 'احتمال الانخفاض' : 'Decline probability'}</small><strong>${pct(analysis.risk_probability_pct)}</strong><span>7 days</span></article>
          <article class="kpi-card"><small>${ar ? 'التغير الفعلي آخر 7 أيام' : 'Observed last-7-day change'}</small><strong class="${number(analysis.observed_change_pct) < 0 ? 'negative' : 'positive'}">${pct(analysis.observed_change_pct, true)}</strong><span>${money(analysis.observed_current_7d_sales)}</span></article>
          <article class="kpi-card"><small>${ar ? 'الانخفاض المتوقع' : 'Forecast decline'}</small><strong>${pct(analysis.predicted_decline_pct)}</strong><span>${pct(analysis.predicted_change_pct, true)}</span></article>
          <article class="kpi-card featured"><small>${ar ? 'دقة التوقع 1−WAPE' : 'Forecast accuracy 1−WAPE'}</small><strong>${pct(analysis.forecast_accuracy_pct)}</strong><span>${escapeHtml(analysis.point_model_name || '')}</span></article>
          <article class="kpi-card"><small>${ar ? 'نسبة الخطأ WAPE' : 'WAPE error'}</small><strong>${pct(analysis.forecast_error_pct)}</strong><span>MAE ${money(analysis.forecast_mae)}</span></article>
          <article class="kpi-card"><small>${ar ? 'تغطية نطاق التوقع' : 'Interval coverage'}</small><strong>${pct(analysis.interval_coverage_pct)}</strong><span>${escapeHtml(analysis.model_version || '')}</span></article>
        </div>
        <article class="panel">
          <div class="panel-heading"><div><h2>${ar ? 'التوصية' : 'Recommendation'}</h2><p>${escapeHtml(recommendation || '')}</p></div></div>
          <div class="table-wrap"><table><thead><tr><th>${ar ? 'التاريخ' : 'Date'}</th><th>${ar ? 'المتوقع' : 'Forecast'}</th><th>${ar ? 'الأدنى' : 'Lower'}</th><th>${ar ? 'الأعلى' : 'Upper'}</th><th>${ar ? 'احتمال الانخفاض' : 'Decline probability'}</th></tr></thead><tbody>${rows}</tbody></table></div>
        </article>
      </section>`;
    slot.hidden = false;
  };

  document.querySelectorAll('[data-browser-analysis-slot]').forEach(slot => renderBrowserSnapshot(slot, latestAnalysis));

  const makeReportCsv = (analysis) => {
    const lines = [];
    const quote = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
    lines.push(['metric', 'value'].map(quote).join(','));
    [
      ['source_file', analysis.source_filename],
      ['data_start', analysis.data_start],
      ['data_end', analysis.data_end],
      ['history_days', analysis.history_days],
      ['decline_probability_pct', number(analysis.risk_probability_pct).toFixed(4)],
      ['observed_change_pct', number(analysis.observed_change_pct).toFixed(4)],
      ['observed_decline_pct', number(analysis.observed_decline_pct).toFixed(4)],
      ['predicted_change_pct', number(analysis.predicted_change_pct).toFixed(4)],
      ['predicted_decline_pct', number(analysis.predicted_decline_pct).toFixed(4)],
      ['forecast_accuracy_1_minus_wape_pct', number(analysis.forecast_accuracy_pct).toFixed(4)],
      ['forecast_error_wape_pct', number(analysis.forecast_error_pct).toFixed(4)],
      ['interval_coverage_pct', number(analysis.interval_coverage_pct).toFixed(4)],
      ['decline_diagnostic_accuracy_pct', analysis.decline_diagnostic_accuracy_pct ?? ''],
      ['decline_diagnostic_error_pct', analysis.decline_diagnostic_error_pct ?? ''],
      ['model_name', analysis.model_name],
      ['model_version', analysis.model_version],
      ['point_model_name', analysis.point_model_name],
      ['point_model_version', analysis.point_model_version],
      ['alert', analysis.alert],
      ['alert_source', analysis.alert_source],
      ['severity', analysis.severity],
    ].forEach(row => lines.push(row.map(quote).join(',')));
    lines.push('');
    lines.push(['date','predicted_sales','lower_bound','upper_bound','decline_probability_pct','decline_percent_pct'].map(quote).join(','));
    (analysis.forecasts || []).forEach(item => lines.push([
      item.date, item.predicted, item.lower, item.upper, item.decline_probability_pct, item.decline_percent_pct,
    ].map(quote).join(',')));
    return '\ufeff' + lines.join('\r\n');
  };

  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-download-analysis]');
    if (!button || !latestAnalysis) return;
    event.preventDefault();
    const blob = new Blob([makeReportCsv(latestAnalysis)], {type: 'text/csv;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    const stamp = String(latestAnalysis.data_end || 'latest').replaceAll(/[^0-9A-Za-z_-]/g, '-');
    anchor.href = url;
    anchor.download = `sales-sentinel-analysis-${stamp}.csv`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  });

  const chart = document.querySelector('.chart[data-sales]');
  if (chart) {
    const sales = parseJson(chart.dataset.sales || '[]', []);
    const forecastSeries = parseJson(chart.dataset.forecasts || '[]', []);
    const values = [...sales.map(item => Number(item.value)), ...forecastSeries.map(item => Number(item.median))];
    const max = Math.max(...values, 1);
    chart.innerHTML = [
      ...sales.map(item => ({...item, kind: 'actual'})),
      ...forecastSeries.map(item => ({date: item.date, value: item.median, kind: 'forecast'})),
    ].map(item => {
      const height = Math.max(4, Number(item.value) / max * 100);
      const value = Number(item.value || 0);
      return `<div class="chart-bar ${item.kind === 'forecast' ? 'forecast' : ''}" style="height:${height}%" title="${escapeHtml(item.date)}: ${value.toLocaleString()}"></div>`;
    }).join('');
  }
})();
