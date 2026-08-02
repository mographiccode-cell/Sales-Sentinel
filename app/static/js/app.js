(() => {
  const chart = document.querySelector('.chart[data-sales]');
  if (!chart) return;
  const sales = JSON.parse(chart.dataset.sales || '[]');
  const forecast = JSON.parse(chart.dataset.forecasts || '[]');
  const values = [...sales.map(item => Number(item.value)), ...forecast.map(item => Number(item.median))];
  const max = Math.max(...values, 1);
  chart.innerHTML = [
    ...sales.map(item => ({...item, kind: 'actual'})),
    ...forecast.map(item => ({date: item.date, value: item.median, kind: 'forecast'})),
  ].map(item => {
    const height = Math.max(4, Number(item.value) / max * 100);
    const value = Number(item.value || 0);
    return `<div class="chart-bar ${item.kind === 'forecast' ? 'forecast' : ''}" style="height:${height}%" title="${item.date}: ${value.toLocaleString()}"></div>`;
  }).join('');
})();
