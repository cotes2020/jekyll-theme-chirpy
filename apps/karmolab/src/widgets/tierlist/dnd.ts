// @ts-nocheck
(function () {
    const T = window.Tierlist = window.Tierlist || {};

    const DRAG_THRESHOLD = 5;
    const ROW_ALIGN_TOL = 10;

    function initDnD(root, { onDrop, shouldBlockDragStart }) {
        function getDropTarget(x, y) {
            const zones = root.querySelectorAll('.tl-dropzone, .tl-pool');
            let best = null;
            let bestArea = Infinity;
            for (const zone of zones) {
                const rect = zone.getBoundingClientRect();
                if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                    const area = rect.width * rect.height;
                    if (area < bestArea) {
                        bestArea = area;
                        best = zone;
                    }
                }
            }
            return best;
        }

        function slotAnchor(cards, k, zoneRect) {
            if (k === 0) {
                if (!cards.length) {
                    return { x: zoneRect.left + 36, y: zoneRect.top + 36 };
                }
                const r0 = cards[0].getBoundingClientRect();
                return { x: r0.left - 6, y: r0.top + r0.height / 2 };
            }
            if (k === cards.length) {
                const r = cards[cards.length - 1].getBoundingClientRect();
                return { x: r.right + 6, y: r.top + r.height / 2 };
            }
            const rL = cards[k - 1].getBoundingClientRect();
            const rR = cards[k].getBoundingClientRect();
            const sameRow = Math.abs(rL.top - rR.top) < ROW_ALIGN_TOL;
            if (sameRow) {
                return {
                    x: (rL.right + rR.left) / 2,
                    y: (rL.top + rL.bottom + rR.top + rR.bottom) / 4,
                };
            }
            return { x: rR.left - 6, y: rR.top + rR.height / 2 };
        }

        function getInsertIndex(zone, x, y) {
            if (zone.dataset.tocDrop === '1') return 999999;
            const cards = Array.from(zone.querySelectorAll('.tl-item:not(.dragging)'));
            if (cards.length === 0) return 0;

            const pad = 3;
            for (let i = 0; i < cards.length; i++) {
                const r = cards[i].getBoundingClientRect();
                if (x >= r.left - pad && x <= r.right + pad && y >= r.top - pad && y <= r.bottom + pad) {
                    const before = x < r.left + r.width * 0.5;
                    return before ? i : i + 1;
                }
            }

            const zr = zone.getBoundingClientRect();
            let bestK = 0;
            let bestD = Infinity;
            for (let k = 0; k <= cards.length; k++) {
                const p = slotAnchor(cards, k, zr);
                const d = Math.hypot(x - p.x, y - p.y);
                if (d < bestD) {
                    bestD = d;
                    bestK = k;
                }
            }
            return bestK;
        }

        function placeholderAlreadyAt(zone, placeholder, idx) {
            const cards = Array.from(zone.querySelectorAll('.tl-item:not(.dragging)'));
            const refChild = cards[idx] ?? null;
            if (placeholder.parentNode !== zone) return false;
            if (refChild === null) {
                const last = cards[cards.length - 1];
                if (!last) return zone.lastElementChild === placeholder;
                return placeholder.previousElementSibling === last && !placeholder.nextElementSibling;
            }
            return placeholder.nextElementSibling === refChild;
        }

        function onPointerDown(e) {
            const itemEl = e.target.closest('.tl-item');
            if (!itemEl || e.button === 2) return;
            const itemId = itemEl.dataset.itemId;
            if (!itemId) return;
            if (shouldBlockDragStart?.(e)) return;

            e.preventDefault();
            itemEl.setPointerCapture(e.pointerId);

            const rect = itemEl.getBoundingClientRect();
            const offsetX = e.clientX - rect.left;
            const offsetY = e.clientY - rect.top;

            let moved = false;
            let dragDone = false;
            let ghost = null;
            let placeholder = null;
            let currentZone = null;
            let lastX = e.clientX;
            let lastY = e.clientY;

            function endDragVisuals() {
                ghost?.remove();
                ghost = null;
                placeholder?.remove();
                placeholder = null;
                itemEl.classList.remove('dragging');
                currentZone?.classList.remove('drag-over');
                currentZone = null;
            }

            function applyDropAt(clientX, clientY) {
                const zone = getDropTarget(clientX, clientY);
                if (zone) {
                    const tierId = zone.dataset.tierId;
                    const idx = getInsertIndex(zone, clientX, clientY);
                    onDrop?.({ itemId, tierId, insertIdx: idx });
                }
            }

            function finishOnce(evClientX, evClientY) {
                if (dragDone) return;
                dragDone = true;
                document.removeEventListener('pointermove', onDocMove, true);
                document.removeEventListener('pointerup', onDocUp, true);
                document.removeEventListener('pointercancel', onDocUp, true);
                itemEl.removeEventListener('lostpointercapture', onLostCapture);
                try {
                    itemEl.releasePointerCapture(e.pointerId);
                } catch (_) {
                    /* noop */
                }

                if (!moved) return;

                endDragVisuals();
                applyDropAt(evClientX, evClientY);
            }

            function onLostCapture() {
                if (!moved || dragDone) return;
                endDragVisuals();
                dragDone = true;
                document.removeEventListener('pointermove', onDocMove, true);
                document.removeEventListener('pointerup', onDocUp, true);
                document.removeEventListener('pointercancel', onDocUp, true);
                applyDropAt(lastX, lastY);
            }

            function applyHover(clientX, clientY) {
                const zone = getDropTarget(clientX, clientY);
                if (currentZone && currentZone !== zone) currentZone.classList.remove('drag-over');
                if (zone) {
                    zone.classList.add('drag-over');
                    currentZone = zone;
                    if (zone.dataset.tocDrop === '1') {
                        if (placeholder?.parentNode) placeholder.remove();
                        return;
                    }
                    const idx = getInsertIndex(zone, clientX, clientY);
                    if (placeholderAlreadyAt(zone, placeholder, idx)) return;
                    if (placeholder.parentNode) placeholder.remove();
                    const cards = zone.querySelectorAll('.tl-item:not(.dragging)');
                    const refChild = cards[idx] ?? null;
                    zone.insertBefore(placeholder, refChild);
                } else if (currentZone) {
                    currentZone.classList.remove('drag-over');
                    currentZone = null;
                    if (placeholder.parentNode) placeholder.remove();
                }
            }

            function onDocMove(ev) {
                lastX = ev.clientX;
                lastY = ev.clientY;

                if (!moved) {
                    if (
                        Math.abs(ev.clientX - e.clientX) < DRAG_THRESHOLD &&
                        Math.abs(ev.clientY - e.clientY) < DRAG_THRESHOLD
                    ) {
                        return;
                    }
                    moved = true;
                    ghost = itemEl.cloneNode(true);
                    ghost.className = 'tl-drag-ghost';
                    ghost.style.width = rect.width + 'px';
                    ghost.style.height = rect.height + 'px';
                    document.body.appendChild(ghost);

                    placeholder = document.createElement('div');
                    placeholder.className = 'tl-placeholder';
                    itemEl.classList.add('dragging');
                }

                ev.preventDefault();
                ghost.style.left = ev.clientX - offsetX + 'px';
                ghost.style.top = ev.clientY - offsetY + 'px';
                applyHover(ev.clientX, ev.clientY);
            }

            function onDocUp(ev) {
                finishOnce(ev.clientX, ev.clientY);
            }

            document.addEventListener('pointermove', onDocMove, true);
            document.addEventListener('pointerup', onDocUp, true);
            document.addEventListener('pointercancel', onDocUp, true);
            itemEl.addEventListener('lostpointercapture', onLostCapture);
        }

        root.addEventListener('pointerdown', onPointerDown);
    }

    T.dnd = { initDnD };
})();
