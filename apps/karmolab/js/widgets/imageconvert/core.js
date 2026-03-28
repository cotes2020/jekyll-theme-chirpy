/**
 * KarmoLab — 이미지 변환 공통 로직 (Canvas 기반, 서버 없음)
 * 다른 스크립트에서 window.KarmoLabImageConvert 로 사용
 */
(function (global) {
    'use strict';

    var MIME_PNG = 'image/png';
    var MIME_JPEG = 'image/jpeg';
    var MIME_WEBP = 'image/webp';

    var RASTER_TYPES = /^image\/(png|jpe?g|pjpeg|webp|gif|bmp|x-ms-bmp)$/i;

    function extFromMime(mime) {
        if (mime === MIME_JPEG) return 'jpg';
        if (mime === MIME_WEBP) return 'webp';
        return 'png';
    }

    function isRasterImageFile(file) {
        if (!file) return false;
        if (file.type && RASTER_TYPES.test(file.type)) return true;
        var n = (file.name || '').toLowerCase();
        return /\.(png|jpe?g|webp|gif|bmp)$/i.test(n);
    }

    function isSvgFile(file) {
        if (!file) return false;
        if (/^image\/svg\+xml$/i.test(file.type)) return true;
        return /\.svgz?$/i.test(file.name || '');
    }

    var webpOutputSupported = null;
    function supportsWebpOutput() {
        if (webpOutputSupported !== null) return webpOutputSupported;
        var c = document.createElement('canvas');
        c.width = 1;
        c.height = 1;
        try {
            var s = c.toDataURL(MIME_WEBP, 0.5);
            webpOutputSupported = s.indexOf('data:image/webp') === 0;
        } catch (_) {
            webpOutputSupported = false;
        }
        return webpOutputSupported;
    }

    /**
     * @param {number} nw
     * @param {number} nh
     * @param {number} maxLong - 긴 변 상한(px), 0이면 리사이즈 안 함
     */
    function computeDimensions(nw, nh, maxLong) {
        if (!nw || !nh) return { w: nw, h: nh };
        var ml = maxLong > 0 ? maxLong : 0;
        if (!ml) return { w: nw, h: nh };
        var longSide = Math.max(nw, nh);
        if (longSide <= ml) return { w: nw, h: nh };
        var scale = ml / longSide;
        return {
            w: Math.max(1, Math.round(nw * scale)),
            h: Math.max(1, Math.round(nh * scale))
        };
    }

    /**
     * @param {HTMLImageElement} img
     * @param {{
     *   outputMime: string,
     *   quality?: number,
     *   maxLongSide?: number,
     *   background?: string,
     *   fillAlpha?: boolean,
     *   smoothing?: 'low'|'medium'|'high'
     *   low = 스무딩 끔(최근접·빠름·계단 느낌). medium/high = 스무딩 켬(브라우저 품질 힌트; 차이는 미미할 수 있음)
     * }} opts
     */
    function imageToCanvas(img, opts) {
        var nw = img.naturalWidth;
        var nh = img.naturalHeight;
        if (!nw || !nh) return null;

        var dim = computeDimensions(nw, nh, opts.maxLongSide || 0);
        var w = dim.w;
        var h = dim.h;

        var canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        var ctx = canvas.getContext('2d');
        if (!ctx) return null;

        var smooth = opts.smoothing || 'high';
        if (smooth === 'low') {
            ctx.imageSmoothingEnabled = false;
        } else {
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = smooth === 'medium' ? 'medium' : 'high';
        }

        var mime = opts.outputMime || MIME_PNG;
        var jpeg = mime === MIME_JPEG;
        var fillAlpha = !!opts.fillAlpha;
        var bg = opts.background || '#ffffff';

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

    function canvasToBlob(canvas, mime, quality) {
        return new Promise(function (resolve, reject) {
            var q;
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

    function convertImage(img, opts) {
        var canvas = imageToCanvas(img, opts);
        if (!canvas) return Promise.reject(new Error('canvas'));
        return canvasToBlob(canvas, opts.outputMime, opts.quality);
    }

    function loadImageFromFile(file) {
        return new Promise(function (resolve, reject) {
            if (isSvgFile(file)) {
                reject(new Error('SVG는 이 도구에서 지원하지 않아요.'));
                return;
            }
            if (!isRasterImageFile(file)) {
                reject(new Error('지원하는 이미지(PNG, JPEG, WebP, GIF, BMP)만 넣어 주세요.'));
                return;
            }
            var url = URL.createObjectURL(file);
            var im = new Image();
            im.onload = function () {
                resolve({ img: im, objectUrl: url, file: file });
            };
            im.onerror = function () {
                try {
                    URL.revokeObjectURL(url);
                } catch (_) {}
                reject(new Error('이미지를 불러오지 못했어요.'));
            };
            im.src = url;
        });
    }

    function baseNameFromFile(file) {
        var n = (file && file.name) || 'image';
        return n.replace(/\.[^.]+$/, '') || 'image';
    }

    function revokeObjectUrl(u) {
        try {
            if (u) URL.revokeObjectURL(u);
        } catch (_) {}
    }

    global.KarmoLabImageConvert = {
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
})(typeof window !== 'undefined' ? window : this);
