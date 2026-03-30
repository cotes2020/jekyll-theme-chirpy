/**
 * 랜덤 생성기 — 시간·날짜 (generator 전용)
 * randomgen-topics.js 로드 후 RANDOMGEN_TOPICS에 추가됨
 */
(function () {
  const topics = window.RANDOMGEN_TOPICS;
  if (!Array.isArray(topics)) return;

  topics.push(
    {
      id: 'date',
      label: '날짜',
      group: '시간',
      generator: function () {
        return String(Math.floor(Math.random() * 31) + 1) + '일';
      }
    },
    {
      id: 'time_24h',
      label: '시간 (24h)',
      group: '시간',
      generator: function () {
        return (
          String(Math.floor(Math.random() * 24)).padStart(2, '0') +
          ':' +
          String(Math.floor(Math.random() * 60)).padStart(2, '0')
        );
      }
    }
  );
})();
