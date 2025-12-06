(function () {
  if (typeof window === 'undefined') return;

  var MERMAID_CDN = 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js';
  var COLOR_SCHEME_ATTR = 'data-user-color-scheme';
  var DEFAULT_COLOR_SCHEME_ATTR = 'data-default-color-scheme';
  var mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

  function cssVar(name, fallback) {
    var raw = getComputedStyle(document.documentElement).getPropertyValue(name);
    return raw ? raw.trim() || fallback : fallback;
  }

  function buildConfig() {
    var palette = {
      nodeBg: cssVar('--mermaid-node-bg', '#fffdf5'),
      nodeBorder: cssVar('--mermaid-node-border', '#d6c39b'),
      nodeText: cssVar('--mermaid-node-text', '#1f2937'),
      highlightBg: cssVar('--mermaid-highlight-bg', '#ffe08a'),
      highlightBorder: cssVar('--mermaid-highlight-border', '#d49c00'),
      secondaryBg: cssVar('--mermaid-secondary-bg', '#d4ebff'),
      secondaryBorder: cssVar('--mermaid-secondary-border', '#5dade2'),
      clusterBg: cssVar('--mermaid-cluster-bg', '#f5f1e3'),
      clusterBorder: cssVar('--mermaid-cluster-border', '#d8ceb8'),
      edgeColor: cssVar('--mermaid-edge-color', '#f2990c'),
      edgeLabel: cssVar('--mermaid-edge-label', '#3b4255'),
      edgeLabelStroke: cssVar('--mermaid-edge-label-stroke', 'rgba(255,255,255,0.85)'),
      fontFamily: cssVar('--mermaid-font-family', 'Inter, "PingFang SC", "Helvetica Neue", Arial, sans-serif')
    };

    var themeCSS = `
      .node rect, .node polygon, .node path {
        stroke-width: 2px;
        rx: 16px;
        ry: 16px;
      }
      .cluster rect {
        fill: ${palette.clusterBg} !important;
        stroke: ${palette.clusterBorder} !important;
        stroke-width: 1.5px;
        rx: 22px;
        ry: 22px;
      }
      .node .label text {
        fill: ${palette.nodeText} !important;
      }
      .edgeLabel rect {
        fill: transparent !important;
        stroke: transparent !important;
      }
      .edgeLabel text {
        fill: ${palette.edgeLabel} !important;
        stroke: ${palette.edgeLabelStroke};
        stroke-width: 2px;
      }
      .node.highlight marquee, .node.secondary marquee { color: inherit; }
    `;

    return {
      startOnLoad: true,
      securityLevel: 'loose',
      theme: 'base',
      themeCSS: themeCSS,
      fontFamily: palette.fontFamily,
      themeVariables: {
        fontFamily: palette.fontFamily,
        primaryColor: palette.nodeBg,
        primaryBorderColor: palette.nodeBorder,
        primaryTextColor: palette.nodeText,
        secondaryColor: palette.secondaryBg,
        secondaryBorderColor: palette.secondaryBorder,
        tertiaryColor: palette.highlightBg,
        tertiaryBorderColor: palette.highlightBorder,
        lineColor: palette.edgeColor,
        edgeColor: palette.edgeColor,
        edgeLabelBackground: 'transparent',
        clusterBkg: palette.clusterBg,
        clusterBorder: palette.clusterBorder,
        noteBkgColor: palette.highlightBg,
        noteTextColor: palette.nodeText,
        background: cssVar('--mermaid-canvas', '#0f172a')
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis',
        diagramPadding: 12,
        nodeSpacing: 65,
        rankSpacing: 85
      },
      sequence: {
        useMaxWidth: true,
        actorFontSize: 16,
        noteFontSize: 14,
        messageFontSize: 15
      },
      gantt: { barHeight: 32 }
    };
  }

  function convertPreToDiv(container) {
    var preBlocks = container.querySelectorAll('pre.mermaid');
    preBlocks.forEach(function (pre) {
      var div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = pre.textContent;
      pre.parentNode.replaceChild(div, pre);
    });
  }

  function renderMermaid() {
    convertPreToDiv(document);
    if (window.mermaid) {
      window.mermaid.initialize(buildConfig());
      window.mermaid.run();
    }
  }

  function initMermaid() {
    renderMermaid();
    document.addEventListener('pjax:complete', renderMermaid);
    observeColorScheme();
  }

  function loadMermaid() {
    return new Promise(function (resolve, reject) {
      if (window.mermaid) {
        resolve(window.mermaid);
        return;
      }
      var script = document.createElement('script');
      script.src = MERMAID_CDN;
      script.async = true;
      script.onload = function () {
        resolve(window.mermaid);
      };
      script.onerror = function (err) {
        console.error('Failed to load Mermaid:', err);
        reject(err);
      };
      document.head.appendChild(script);
    });
  }

  function observeColorScheme() {
    var root = document.documentElement;
    var observer = new MutationObserver(function (mutations) {
      if (mutations.some(function (m) { return m.attributeName === COLOR_SCHEME_ATTR; })) {
        renderMermaid();
      }
    });
    observer.observe(root, { attributes: true, attributeFilter: [COLOR_SCHEME_ATTR] });

    // 当用户未手动切换而系统主题改变时（data-user-color-scheme 为空）
    mediaQuery.addEventListener('change', function () {
      var hasUserSetting = root.hasAttribute(COLOR_SCHEME_ATTR);
      if (!hasUserSetting || root.getAttribute(COLOR_SCHEME_ATTR) === '') {
        renderMermaid();
      }
    });

    // 默认模式为 auto 时也需要监听属性更新
    if (root.getAttribute(DEFAULT_COLOR_SCHEME_ATTR) === 'auto' && !root.hasAttribute(COLOR_SCHEME_ATTR)) {
      renderMermaid();
    }
  }

  loadMermaid().then(initMermaid).catch(function () {
    console.error('Mermaid diagrams failed to initialize.');
  });
})();
