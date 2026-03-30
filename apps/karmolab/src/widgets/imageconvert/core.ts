/**
 * KarmoLab — 이미지 변환 공통 로직 (Canvas 기반, 서버 없음)
 * 다른 스크립트에서 window.KarmoLabImageConvert 로 사용
 */
import type { ImageConvertOptions, KarmoLabImageConvertAPI } from '../../../types/karmolab';

(function (global: Window) {
  const MIME_PNG = 'image/png';
  const MIME_JPEG = 'image/jpeg';
  const MIME_WEBP = 'image/webp';

  const RASTER_TYPES = /^image\/(png|jpe?g|pjpeg|webp|gif|bmp|x-ms-bmp)$/i;

  function extFromMime(mime: string): string {
    if (mime === MIME_JPEG) return 'jpg';
    if (mime === MIME_WEBP) return 'webp';
    return 'png';
  }

  function isRasterImageFile(file: File): boolean {
    if (!file) return false;
    if (file.type && RASTER_TYPES.test(file.type)) return true;
    const n = (file.name || '').toLowerCase();
    return /\.(png|jpe?g|webp|gif|bmp)$/i.test(n);
  }

  function isSvgFile(file: File): boolean {
    if (!file) return false;
    if (/^image\/svg\+xml$/i.test(file.type)) return true;
    return /\.svgz?$/i.test(file.name || '');
  }

  let webpOutputSupported: boolean | null = null;
  function supportsWebpOutput(): boolean {
    if (webpOutputSupported !== null) return webpOutputSupported;
    const c = document.createElement('canvas');
    c.width = 1;
    c.height = 1;
    try {
      const s = c.toDataURL(MIME_WEBP, 0.5);
      webpOutputSupported = s.indexOf('data:image/webp') === 0;
    } catch {
      webpOutputSupported = false;
    }
    return webpOutputSupported;
  }

  function computeDimensions(nw: number, nh: number, maxLong: number): { w: number; h: number } {
    if (!nw || !nh) return { w: nw, h: nh };
    const ml = maxLong > 0 ? maxLong : 0;
    if (!ml) return { w: nw, h: nh };
    const longSide = Math.max(nw, nh);
    if (longSide <= ml) return { w: nw, h: nh };
    const scale = ml / longSide;
    return {
      w: Math.max(1, Math.round(nw * scale)),
      h: Math.max(1, Math.round(nh * scale))
    };
  }

  function imageToCanvas(img: HTMLImageElement, opts: ImageConvertOptions): HTMLCanvasElement | null {
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    if (!nw || !nh) return null;

    const dim = computeDimensions(nw, nh, opts.maxLongSide || 0);
    const w = dim.w;
    const h = dim.h;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const smooth = opts.smoothing || 'high';
    if (smooth === 'low') {
      ctx.imageSmoothingEnabled = false;
    } else {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = smooth === 'medium' ? 'medium' : 'high';
    }

    const mime = opts.outputMime || MIME_PNG;
    const jpeg = mime === MIME_JPEG;
    const fillAlpha = !!opts.fillAlpha;
    const bg = opts.background || '#ffffff';

    if (jpeg) {
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);
    } else if (fillAlpha && bg && bg !== 'transparent') {
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);
    }

    ctx.drawImage(img, 0, 0, nw, nh, 0, 0, w, h);
    return canvas;
  }

  function canvasToBlob(canvas: HTMLCanvasElement, mime: string, quality?: number): Promise<Blob> {
    return new Promise(function (resolve, reject) {
      let q: number | undefined;
      if (mime === MIME_PNG) q = undefined;
      else q = Math.min(1, Math.max(0.05, quality == null ? 0.92 : quality));
      canvas.toBlob(
        function (b) {
          if (b) resolve(b);
          else reject(new Error('toBlob failed'));
        },
        mime,
        q
      );
    });
  }

  function convertImage(img: HTMLImageElement, opts: ImageConvertOptions): Promise<Blob> {
    const canvas = imageToCanvas(img, opts);
    if (!canvas) return Promise.reject(new Error('canvas'));
    return canvasToBlob(canvas, opts.outputMime, opts.quality);
  }

  function loadImageFromFile(file: File): Promise<{
    img: HTMLImageElement;
    objectUrl: string;
    file: File;
  }> {
    return new Promise(function (resolve, reject) {
      if (isSvgFile(file)) {
        reject(new Error('SVG는 이 도구에서 지원하지 않아요.'));
        return;
      }
      if (!isRasterImageFile(file)) {
        reject(new Error('지원하는 이미지(PNG, JPEG, WebP, GIF, BMP)만 넣어 주세요.'));
        return;
      }
      const url = URL.createObjectURL(file);
      const im = new Image();
      im.onload = function () {
        resolve({ img: im, objectUrl: url, file: file });
      };
      im.onerror = function () {
        try {
          URL.revokeObjectURL(url);
        } catch {
          /* noop */
        }
        reject(new Error('이미지를 불러오지 못했어요.'));
      };
      im.src = url;
    });
  }

  function baseNameFromFile(file: File): string {
    const n = (file && file.name) || 'image';
    return n.replace(/\.[^.]+$/, '') || 'image';
  }

  function revokeObjectUrl(u: string | undefined): void {
    try {
      if (u) URL.revokeObjectURL(u);
    } catch {
      /* noop */
    }
  }

  const api: KarmoLabImageConvertAPI = {
    MIME_PNG: MIME_PNG,
    MIME_JPEG: MIME_JPEG,
    MIME_WEBP: MIME_WEBP,
    extFromMime: extFromMime,
    isRasterImageFile: isRasterImageFile,
    isSvgFile: isSvgFile,
    supportsWebpOutput: supportsWebpOutput,
    computeDimensions: computeDimensions,
    imageToCanvas: imageToCanvas,
    canvasToBlob: canvasToBlob,
    convertImage: convertImage,
    loadImageFromFile: loadImageFromFile,
    baseNameFromFile: baseNameFromFile,
    revokeObjectUrl: revokeObjectUrl
  };

  global.KarmoLabImageConvert = api;
})(typeof window !== 'undefined' ? window : (globalThis as unknown as Window));
