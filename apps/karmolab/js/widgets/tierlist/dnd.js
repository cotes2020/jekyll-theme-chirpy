(function () {
    const T = window.Tierlist = window.Tierlist || {};

    function initDnD(rootEl, { onDrop }) {
        function getDropTarget(x, y) {
            const zones = rootEl.querySelectorAll('.tl-dropzone, .tl-pool');
            let best = null, bestDist = Infinity;
            for (const zone of zones) {
                const rect = zone.getBoundingClientRect();
                if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                    const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
                    const dist = Math.hypot(x - cx, y - cy);
                    if (dist < bestDist) { bestDist = dist; best = zone; }
                }
            }
            return best;
        }

        function getInsertIndex(zone, x) {
            const cards = Array.from(zone.querySelectorAll('.tl-item:not(.dragging)'));
            for (let i = 0; i < cards.length; i++) {
                const rect = cards[i].getBoundingClientRect();
                if (x < rect.left + rect.width / 2) return i;
            }
            return cards.length;
        }

        function onPointerDown(e) {
            const itemEl = e.target.closest('.tl-item');
            if (!itemEl || e.button === 2) return;
            const itemId = itemEl.dataset.itemId;
            if (!itemId) return;

            e.preventDefault();
            itemEl.setPointerCapture(e.pointerId);

            const rect = itemEl.getBoundingClientRect();
            const offsetX = e.clientX - rect.left;
            const offsetY = e.clientY - rect.top;

            let moved = false;
            let ghost = null;
            let placeholder = null;
            let currentZone = null;

            const onMove = (ev) => {
                if (!moved) {
                    if (Math.abs(ev.clientX - e.clientX) < 4 && Math.abs(ev.clientY - e.clientY) < 4) return;
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

                ghost.style.left = (ev.clientX - offsetX) + 'px';
                ghost.style.top = (ev.clientY - offsetY) + 'px';

                const zone = getDropTarget(ev.clientX, ev.clientY);
                if (currentZone && currentZone !== zone) currentZone.classList.remove('drag-over');
                if (zone) {
                    zone.classList.add('drag-over');
                    currentZone = zone;
                    const idx = getInsertIndex(zone, ev.clientX);
                    const children = Array.from(zone.querySelectorAll('.tl-item:not(.dragging), .tl-placeholder'));
                    const placeholderIdx = children.indexOf(placeholder);
                    if (placeholderIdx !== -1 && (placeholderIdx === idx || placeholderIdx === idx - 1)) return;
                    if (placeholder.parentNode) placeholder.remove();
                    const refChild = zone.querySelectorAll('.tl-item:not(.dragging)')[idx];
                    zone.insertBefore(placeholder, refChild || null);
                }
            };

            const onUp = (ev) => {
                itemEl.releasePointerCapture(ev.pointerId);
                itemEl.removeEventListener('pointermove', onMove);
                itemEl.removeEventListener('pointerup', onUp);

                if (!moved) return;

                ghost?.remove();
                placeholder?.remove();
                itemEl.classList.remove('dragging');
                currentZone?.classList.remove('drag-over');

                const zone = getDropTarget(ev.clientX, ev.clientY);
                if (zone) {
                    const tierId = zone.dataset.tierId;
                    const idx = getInsertIndex(zone, ev.clientX);
                    onDrop?.({ itemId, tierId, insertIdx: idx });
                }
            };

            itemEl.addEventListener('pointermove', onMove);
            itemEl.addEventListener('pointerup', onUp);
        }

        rootEl.addEventListener('pointerdown', onPointerDown);
    }

    T.dnd = { initDnD };
})();

