import cheerio from 'cheerio';

// Node 18+ 에서는 글로벌 fetch가 있으므로 우선 사용하고,
// 없는 환경에서는 node-fetch를 동적 import로 로드한다.
let fetchImpl: typeof fetch | null = typeof globalThis.fetch === 'function' ? globalThis.fetch.bind(globalThis) : null;

async function ensureFetch(): Promise<typeof fetch> {
  if (fetchImpl) return fetchImpl;
  const mod: any = await import('node-fetch');
  fetchImpl = (mod.default || mod).bind(mod);
  return fetchImpl!;
}

const PUBLISHER_SALE_URL = 'https://assetstore.unity.com/ko-KR/publisher-sale';

function normalizeAssetStoreHref(href: string): string {
  return href.startsWith('http') ? href : `https://assetstore.unity.com${href}`;
}

/** 선물 CTA 문구(본문·접근성 라벨·툴팁 등)에 무료 에셋 링크가 있는지 */
function textLooksLikeGiftCta(blob: string): boolean {
  const lower = blob.toLowerCase();
  if (/get your (free )?gift/.test(lower)) return true;
  if (/\bfree gift\b/.test(lower)) return true;
  // 한국어 페이지 대비
  if (/무료\s*선물|선물\s*받기|무료\s*받기/.test(blob)) return true;
  return false;
}

/**
 * HTML이 바뀌어도 선물 버튼은 보통 /packages/ + aria-label(gift) 조합으로 남는다.
 */
function extractGiftPackageHrefFromRawHtml(html: string): string | null {
  const hrefFirst = /<a\b[^>]*\bhref="(\/packages\/[^"]+)"[^>]*\baria-label="[^"]*(?:gift|선물)[^"]*"/i;
  const ariaFirst = /<a\b[^>]*\baria-label="[^"]*(?:gift|선물)[^"]*"[^>]*\bhref="(\/packages\/[^"]+)"/i;
  const m = html.match(hrefFirst) || html.match(ariaFirst);
  return m ? m[1] : null;
}

function findGiftAssetUrl($: ReturnType<typeof cheerio.load>, html: string): string | null {
  let found: string | null = null;
  $('a[href*="/packages/"]').each((_, el) => {
    if (found) return;
    const $a = $(el);
    const href = $a.attr('href');
    if (!href || !href.includes('/packages/')) return;
    const blob = [
      $a.text(),
      $a.attr('aria-label') ?? '',
      $a.attr('title') ?? '',
    ].join(' ');
    if (textLooksLikeGiftCta(blob)) {
      found = normalizeAssetStoreHref(href);
    }
  });
  if (found) return found;

  const fallback = extractGiftPackageHrefFromRawHtml(html);
  return fallback ? normalizeAssetStoreHref(fallback) : null;
}

export type UnityFreeAssetInfo = {
  couponCode: string | null;
  assetName: string | null;
  assetUrl: string | null;
  promoText: string | null;
  imageUrl: string | null;
};

/**
 * Unity Publisher Sale 페이지에서 현재 무료 에셋 정보를 파싱합니다.
 * - 쿠폰 코드 (예: SURIYUN2026)
 * - 에셋 이름
 * - 에셋 상세 페이지 링크
 * - 안내 문구/기간 텍스트
 */
export async function fetchUnityFreeAssetInfo(): Promise<UnityFreeAssetInfo | null> {
  const fetch = await ensureFetch();
  const res = await fetch(PUBLISHER_SALE_URL, {
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36',
      'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    },
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

  // 무료 에셋 상세 링크: 본문이 비어 있고 aria-label만 있는 경우가 있어 DOM + 원본 HTML 폴백
  const assetUrl = findGiftAssetUrl($, html);

  // 기간/안내 문구(“promotion end …” 같은 문자열) 추출
  let promoText: string | null = null;
  const promoMatch = bodyText.match(/promotion end[s]?\s+[^.]+?\./i);
  if (promoMatch) {
    promoText = promoMatch[0].trim();
  }

  if (!couponCode && !assetUrl) {
    return null;
  }

  const imageUrl = assetUrl ? await fetchUnityAssetImage(assetUrl) : null;

  return {
    couponCode,
    assetName,
    assetUrl,
    promoText,
    imageUrl,
  };
}

/**
 * 에셋 상세 페이지에서 대표 이미지(og:image)를 가져옵니다.
 */
async function fetchUnityAssetImage(assetUrl: string): Promise<string | null> {
  try {
    const fetch = await ensureFetch();
    const res = await fetch(assetUrl, {
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36',
      },
    });
    if (!res.ok) return null;

    const html = await res.text();
    const $ = cheerio.load(html);

    const ogImage = $('meta[property="og:image"]').attr('content');
    if (ogImage) {
      try {
        return new URL(ogImage, assetUrl).href;
      } catch {
        return ogImage.startsWith('http') ? ogImage : null;
      }
    }

    const firstImg = $('img').first().attr('src');
    if (firstImg) {
      try {
        return new URL(firstImg, assetUrl).href;
      } catch {
        return firstImg.startsWith('http') ? firstImg : null;
      }
    }
    return null;
  } catch {
    return null;
  }
}

