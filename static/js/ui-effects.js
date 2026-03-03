(function () {
    function updateCameraBadgeState() {
        const badge = document.getElementById('cameraStatusBadge');
        const textNode = document.getElementById('cameraStatusText');
        if (!badge || !textNode) {
            return;
        }

        const text = (textNode.textContent || '').toLowerCase();
        badge.classList.remove('state-waiting', 'state-active');

        if (text.includes('active')) {
            badge.classList.add('state-active');
        } else {
            badge.classList.add('state-waiting');
        }
    }

    function enhanceCardHover() {
        const cards = document.querySelectorAll('.glass-card, .glass-stat, .camera-container');
        cards.forEach((card) => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-2px)';
            });
            card.addEventListener('mouseleave', () => {
                card.style.transform = '';
            });
        });
    }

    function bindProgressAnimation() {
        const progressBar = document.getElementById('trainProgressBar');
        if (!progressBar) {
            return;
        }

        const observer = new MutationObserver(() => {
            const widthValue = Number((progressBar.style.width || '0').replace('%', ''));
            progressBar.style.filter = widthValue > 0 ? 'drop-shadow(0 0 10px rgba(99, 102, 241, 0.45))' : '';
        });

        observer.observe(progressBar, {
            attributes: true,
            attributeFilter: ['style']
        });
    }

    window.addEventListener('DOMContentLoaded', () => {
        updateCameraBadgeState();
        enhanceCardHover();
        bindProgressAnimation();

        const textNode = document.getElementById('cameraStatusText');
        if (textNode) {
            const cameraObserver = new MutationObserver(updateCameraBadgeState);
            cameraObserver.observe(textNode, {
                childList: true,
                characterData: true,
                subtree: true
            });
        }
    });
})();
