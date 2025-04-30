document.addEventListener('DOMContentLoaded', function () {
  const cal = new CalHeatmap();

  // 히트맵 렌더링 함수 정의
  const renderHeatmap = () => {
    cal.paint(
      {
        itemSelector: '#posts-heatmap',

        data: {
          source: '/assets/js/data/posts-heatmap.json',
          type: 'json',
          x: 'date',
          y: (d) => d['value']
        },

        range: 3,

        domain: {
          type: 'month',
          gutter: 8,

          label: {
            text: 'MMM',
            textAlign: 'left',
            position: 'bottom'
          }
        },

        subDomain: {
          type: 'ghDay',
          radius: 3,
          width: 11,
          height: 11,
          gutter: 4
        },

        date: {
          start: new Date(new Date().setMonth(new Date().getMonth() - 2))
        },

        scale: {
          color: {
            type: 'threshold',
            range: [
              getComputedStyle(document.documentElement)
                .getPropertyValue('--heatmap-cell-1')
                .trim(),
              getComputedStyle(document.documentElement)
                .getPropertyValue('--heatmap-cell-2')
                .trim(),
              getComputedStyle(document.documentElement)
                .getPropertyValue('--heatmap-cell-3')
                .trim(),
              getComputedStyle(document.documentElement)
                .getPropertyValue('--heatmap-cell-4')
                .trim()
            ],
            interpolate: 'rgb',
            domain: [1, 2, 3],
            unknown: getComputedStyle(document.documentElement)
              .getPropertyValue('--heatmap-cell-1')
              .trim()
          }
        }
      },
      [
        [
          Tooltip,
          {
            text: function (date, value, dayjsDate) {
              return (
                dayjsDate.format('YYYY-MM-DD') +
                ' | ' +
                (value ? value : '0') +
                ' Posts Uploaded.'
              );
            }
          }
        ]
      ]
    );
  };

  // 페이지 로드 시 최초 렌더링
  renderHeatmap();

  // 모드 전환 버튼 이벤트 등록
  const modeToggleBtn = document.querySelector('#mode-toggle');
  if (modeToggleBtn) {
    modeToggleBtn.addEventListener('click', () => {
      cal.destroy(); // 기존 히트맵 제거
      renderHeatmap(); // 히트맵 다시 그리기
    });
  }
});
