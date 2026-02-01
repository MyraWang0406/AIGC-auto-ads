(function () {
  'use strict';

  const SAMPLES = {
    game: {
      vertical: "game",
      product_name: "《王者荣耀》新赛季",
      target_audience: "18-35岁手游玩家，喜欢MOBA、竞技",
      key_selling_points: ["新英雄「镜」上线", "赛季皮肤免费领", "排位机制优化，上分更容易"],
      tone: "热血、年轻",
      constraints: ["不得出现未成年人游戏画面", "需标注游戏适龄提示"],
      extra_context: "投放平台：抖音、快手；时长15秒以内",
      no_exaggeration: true,
    },
    ecommerce: {
      vertical: "ecommerce",
      product_name: "XXX 氨基酸洗发水",
      target_audience: "25-40岁女性，关注护发、头皮健康",
      key_selling_points: ["无硅油配方", "氨基酸表活，温和不刺激", "控油蓬松，72小时不扁塌"],
      tone: "专业、种草",
      constraints: ["功效宣称需有依据", "不得暗示治疗功效"],
      extra_context: "投放平台：小红书、抖音；促销：首单立减30",
      no_exaggeration: true,
    },
  };

  const cardJsonEl = document.getElementById('cardJson');
  const nEl = document.getElementById('n');
  const btnSubmit = document.getElementById('btnSubmit');
  const apiBaseEl = document.getElementById('apiBase');
  const errorBox = document.getElementById('errorBox');
  const resultSection = document.getElementById('resultSection');
  const tbody = document.querySelector('#resultTable tbody');
  const btnDownloadCsv = document.getElementById('btnDownloadCsv');
  const btnDownloadMd = document.getElementById('btnDownloadMd');

  let lastResult = null;

  // 加载示例
  document.querySelectorAll('[data-sample]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const key = btn.getAttribute('data-sample');
      const sample = SAMPLES[key];
      if (sample) {
        cardJsonEl.value = JSON.stringify(sample, null, 2);
        hideError();
      }
    });
  });

  function hideError() {
    errorBox.classList.add('hidden');
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
  }

  function getApiBase() {
    const v = (apiBaseEl.value || '').trim();
    if (v) return v.replace(/\/$/, '');
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }
    return '';
  }

  btnSubmit.addEventListener('click', async function () {
    hideError();
    btnSubmit.disabled = true;
    try {
      let card;
      try {
        card = JSON.parse(cardJsonEl.value || '{}');
      } catch (e) {
        showError('结构卡片 JSON 格式错误：' + e.message);
        return;
      }
      const n = Math.max(1, Math.min(10, parseInt(nEl.value, 10) || 5));

      const base = getApiBase();
      const url = base ? base + '/generate_and_review' : '/generate_and_review';

      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ card, n }),
      });

      const data = await res.json();

      if (!res.ok) {
        showError(data.detail || data.error || '请求失败');
        return;
      }

      if (!data.ok) {
        showError(data.error || '处理失败');
        return;
      }

      lastResult = data;
      renderTable(data.table);
      resultSection.classList.remove('hidden');
    } catch (e) {
      showError('请求异常：' + e.message + '。请确认 API 地址正确且后端已启动。');
    } finally {
      btnSubmit.disabled = false;
    }
  });

  function renderTable(table) {
    tbody.innerHTML = '';
    if (!table || table.length === 0) return;
    table.forEach(function (row) {
      const tr = document.createElement('tr');
      const cls = 'verdict-' + (row.decision || '');
      tr.innerHTML =
        '<td>' + row.index + '</td>' +
        '<td>' + escapeHtml(row.variant_id || '-') + '</td>' +
        '<td>' + escapeHtml(row.headline || '-') + '</td>' +
        '<td class="' + cls + '">' + escapeHtml(row.decision || '-') + '</td>' +
        '<td>' + escapeHtml(row.fuse_level || '-') + '</td>' +
        '<td>' + escapeHtml(String(row.white_traffic_risk_final ?? '-')) + '</td>' +
        '<td>' + escapeHtml(String(row.clarity ?? '-')) + '</td>' +
        '<td>' + escapeHtml(String(row.hook_strength ?? '-')) + '</td>' +
        '<td>' + escapeHtml(String(row.compliance_safety ?? '-')) + '</td>' +
        '<td>' + escapeHtml((row.summary || '-').slice(0, 60)) + '</td>';
      tbody.appendChild(tr);
    });
  }

  function escapeHtml(s) {
    if (s == null) return '-';
    const div = document.createElement('div');
    div.textContent = String(s);
    return div.innerHTML;
  }

  function download(content, filename, mime) {
    const a = document.createElement('a');
    a.href = 'data:' + mime + ';charset=utf-8,' + encodeURIComponent(content);
    a.download = filename;
    a.click();
  }

  btnDownloadCsv.addEventListener('click', function () {
    if (lastResult && lastResult.csv) {
      download(lastResult.csv, 'creative_review.csv', 'text/csv');
    }
  });

  btnDownloadMd.addEventListener('click', function () {
    if (lastResult && lastResult.markdown) {
      download(lastResult.markdown, 'creative_review.md', 'text/markdown');
    }
  });

  // 本地开发默认 API 地址
  if (window.location.port === '5500' || window.location.hostname === 'localhost') {
    apiBaseEl.placeholder = 'http://localhost:8000（uvicorn 默认端口）';
  }
})();
