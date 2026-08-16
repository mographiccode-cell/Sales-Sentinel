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
    const severity = escapeHtml(analysis.severity || 'medium');
    const severityLabel = ar
      ? ({high: 'مرتفع', medium: 'متوسط', low: 'منخفض'}[analysis.severity] || 'متوسط')
      : String(analysis.severity || 'medium').replace(/^./, c => c.toUpperCase());
    const sourceLabel = ar
      ? ({model_and_observed: 'النموذج + الانخفاض الفعلي', model: 'النموذج', observed: 'الانخفاض الفعلي'}[analysis.alert_source] || analysis.alert_source || '')
      : String(analysis.alert_source || '').replaceAll('_', ' ');
    const recommendation = ar ? analysis.recommendation_ar : analysis.recommendation_en;
    const rows = forecasts.map(item => `
      <tr>
        <td><strong>${escapeHtml(item.date)}</strong></td>
        <td class="forecast-value-v2">${money(item.predicted)}</td>
        <td><span class="range-v2">${money(item.lower)} — ${money(item.upper)}</span></td>
        <td><span class="probability-pill-v2">${pct(item.decline_probability_pct)}</span></td>
        <td><span class="decline-pill-v2">${pct(item.decline_percent_pct)}</span></td>
      </tr>`).join('');

    slot.innerHTML = `
      <section class="analysis-section analysis-section-v2 browser-analysis">
        <div class="result-hero-v2 severity-${severity}">
          <div class="result-copy-v2">
            <div class="result-badges-v2">
              <span class="status-badge-v2">${analysis.alert ? (ar ? `تنبيه ${severityLabel}` : `Alert · ${severityLabel}`) : (ar ? 'لا يوجد تنبيه' : 'No active alert')}</span>
              <span class="source-badge-v2">${escapeHtml(sourceLabel)}</span>
            </div>
            <span class="eyebrow">BROWSER ANALYSIS CACHE</span>
            <h2>${analysis.alert ? (ar ? 'آخر إشارة انخفاض محفوظة في المتصفح' : 'Latest decline signal saved in this browser') : (ar ? 'آخر تحليل محفوظ في المتصفح' : 'Latest analysis saved in this browser')}</h2>
            <p>${escapeHtml(analysis.source_filename || '')} · ${escapeHtml(analysis.data_start || '')} → ${escapeHtml(analysis.data_end || '')}</p>
            <div class="analysis-actions result-actions-v2"><button type="button" class="primary-button" data-download-analysis>${ar ? 'تنزيل التقرير CSV' : 'Download CSV report'}</button></div>
          </div>
          <div class="risk-score-v2">
            <span>${ar ? 'احتمال الانخفاض' : 'Decline probability'}</span>
            <strong>${pct(analysis.risk_probability_pct)}</strong>
            <div class="risk-track-v2"><i style="width:${number(analysis.risk_probability_pct).toFixed(1)}%"></i></div>
            <small>${ar ? 'حد القرار' : 'Decision threshold'} ${pct(analysis.decision_threshold_pct)}</small>
          </div>
        </div>
        <div class="primary-metrics-v2">
          <article><small>${ar ? 'الانخفاض الفعلي' : 'Observed decline'}</small><strong class="negative">${pct(analysis.observed_decline_pct)}</strong><span>${ar ? 'محسوب من البيانات' : 'Calculated from data'}</span></article>
          <article><small>${ar ? 'الانخفاض المتوقع' : 'Forecast decline'}</small><strong class="negative">${pct(analysis.predicted_decline_pct)}</strong><span>${pct(analysis.predicted_change_pct, true)}</span></article>
          <article><small>${ar ? 'إجمالي توقع 7 أيام' : '7-day forecast total'}</small><strong>${money(analysis.forecast_total)}</strong><span>7 ${ar ? 'أيام' : 'days'}</span></article>
          <article class="accuracy"><small>${ar ? 'دقة توقع قيمة المبيعات' : 'Sales-value forecast accuracy'}</small><strong>${pct(analysis.forecast_accuracy_pct)}</strong><span>1 − WAPE</span></article>
        </div>
        <div class="analysis-grid-v2">
          <article class="panel decision-card-v2">
            <div class="decision-title-v2"><span class="decision-icon-v2">AI</span><div><span class="eyebrow">DECISION SUPPORT</span><h2>${ar ? 'القرار والتوصية' : 'Decision and recommendation'}</h2></div></div>
            <p class="decision-summary-v2">${escapeHtml(recommendation || '')}</p>
            <div class="decision-facts-v2"><div><small>${ar ? 'الخطر' : 'Risk'}</small><strong>${pct(analysis.risk_probability_pct)}</strong></div><div><small>${ar ? 'الانخفاض الفعلي' : 'Observed decline'}</small><strong>${pct(analysis.observed_decline_pct)}</strong></div><div><small>${ar ? 'الانخفاض المتوقع' : 'Forecast decline'}</small><strong>${pct(analysis.predicted_decline_pct)}</strong></div></div>
          </article>
          <article class="panel quality-card-v2">
            <div class="panel-heading"><div><span class="eyebrow">MODEL QUALITY</span><h2>${ar ? 'الدقة والخطأ بوضوح' : 'Accuracy and error, separated'}</h2></div></div>
            <div class="quality-meter-v2"><div class="quality-meter-head-v2"><span>${ar ? 'توقع قيمة المبيعات' : 'Sales-value forecast'}</span><strong>${pct(analysis.forecast_accuracy_pct)}</strong></div><div class="meter-track-v2"><i style="width:${number(analysis.forecast_accuracy_pct).toFixed(1)}%"></i></div><div class="quality-sub-v2"><span>WAPE ${pct(analysis.forecast_error_pct)}</span><span>${ar ? 'تغطية النطاق' : 'Coverage'} ${pct(analysis.interval_coverage_pct)}</span></div></div>
            ${analysis.decline_diagnostic_accuracy_pct == null ? '' : `<div class="quality-meter-v2 diagnostic"><div class="quality-meter-head-v2"><span>${ar ? 'تصنيف الانخفاض · دليل خارجي' : 'Decline classification · external evidence'}</span><strong>${pct(analysis.decline_diagnostic_accuracy_pct)}</strong></div><div class="meter-track-v2"><i style="width:${number(analysis.decline_diagnostic_accuracy_pct).toFixed(1)}%"></i></div><div class="quality-sub-v2"><span>${ar ? 'صحيح' : 'Correct'} ${escapeHtml(analysis.decline_correct_count)}/${escapeHtml(analysis.decline_diagnostic_sample_size)}</span><span>${ar ? 'خطأ' : 'Wrong'} ${escapeHtml(analysis.decline_wrong_count)}/${escapeHtml(analysis.decline_diagnostic_sample_size)}</span></div></div>`}
          </article>
        </div>
        <article class="panel analysis-report-panel report-panel-v2">
          <div class="panel-heading"><div><span class="eyebrow">7-DAY REPORT</span><h2>${ar ? 'التوقع اليومي القادم' : 'Upcoming daily forecast'}</h2></div></div>
          <div class="table-wrap"><table class="forecast-table-v2"><thead><tr><th>${ar ? 'التاريخ' : 'Date'}</th><th>${ar ? 'المبيعات المتوقعة' : 'Predicted sales'}</th><th>${ar ? 'نطاق التوقع' : 'Prediction range'}</th><th>${ar ? 'احتمال الانخفاض' : 'Decline probability'}</th><th>${ar ? 'انخفاض القيمة' : 'Value decline'}</th></tr></thead><tbody>${rows}</tbody></table></div>
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
