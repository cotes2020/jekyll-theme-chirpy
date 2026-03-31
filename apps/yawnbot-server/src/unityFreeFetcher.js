// Node 18+ 에서는 글로벌 fetch가 존재하지만, node-fetch를 같이 쓸 수 있도록 안전하게 처리
const _nodeFetch = require('node-fetch');
const fetch = typeof global.fetch === 'function' ? global.fetch.bind(global) : _nodeFetch.default || _nodeFetch;
const cheerio = require('cheerio');

const PUBLISHER_SALE_URL = 'https://assetstore.unity.com/ko-KR/publisher-sale';

/**
 * Unity Publisher Sale 페이지에서 현재 무료 에셋 정보를 파싱합니다.
 * - 쿠폰 코드 (예: SURIYUN2026)
 * - 에셋 이름
 * - 에셋 상세 페이지 링크
 * - 안내 문구/기간 텍스트
 */
async function fetchUnityFreeAssetInfo() {
    const res = await fetch(PUBLISHER_SALE_URL, {
        headers: {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
        }
    });

    if (!res.ok) {
        throw new Error(`Unity Publisher Sale 페이지 요청 실패: ${res.status} ${res.statusText}`);
    }

    const html = await res.text();
    const $ = cheerio.load(html);
    const bodyText = $('body').text().replace(/\s+/g, ' ').trim();

    // "coupon code XXXXX" 패턴에서 코드 추출
    const couponMatch = bodyText.match(/coupon code\s+([A-Z0-9]+)/i);
    const couponCode = couponMatch ? couponMatch[1] : null;

    // "Add XXX to your cart" 부분에서 에셋 이름 추출
    const nameMatch = bodyText.match(/Add\s+(.+?)\s+to your cart/i);
    const assetName = nameMatch ? nameMatch[1].trim() : null;

    // "Get your gift" 링크(버튼)의 href 탐색
    let assetUrl = null;
    $('a').each((_, el) => {
        const text = $(el).text().trim().toLowerCase();
        if (text.includes('get your gift')) {
            const href = $(el).attr('href');
            if (href) {
                assetUrl = href.startsWith('http') ? href : `https://assetstore.unity.com${href}`;
            }
        }
    });

    // 기간/안내 문구(“promotion end …” 같은 문자열) 추출
    let promoText = null;
    const promoMatch = bodyText.match(/promotion end[s]?\s+[^.]+?\./i);
    if (promoMatch) {
        promoText = promoMatch[0].trim();
    }

    if (!couponCode && !assetUrl) {
        // 현재 무료 증정이 없을 수도 있으므로, 에러 대신 null 반환
        return null;
    }

    const imageUrl = assetUrl ? await fetchUnityAssetImage(assetUrl) : null;

    return {
        couponCode,
        assetName,
        assetUrl,
        promoText,
        imageUrl
    };
}

/**
 * 에셋 상세 페이지에서 대표 이미지(og:image)를 가져옵니다.
 */
async function fetchUnityAssetImage(assetUrl) {
    try {
        const res = await fetch(assetUrl, {
            headers: {
                'User-Agent':
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36'
            }
        });
        if (!res.ok) return null;

        const html = await res.text();
        const $ = cheerio.load(html);

        const ogImage = $('meta[property="og:image"]').attr('content');
        if (ogImage) return ogImage;

        // og:image가 없으면 첫 번째 <img> 시도
        const firstImg = $('img').first().attr('src');
        return firstImg || null;
    } catch {
        return null;
    }
}

module.exports = {
    fetchUnityFreeAssetInfo,
};

