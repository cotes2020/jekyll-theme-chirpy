/*
 * Copy code in code blocks to clipboard.
 * Porting from: kitian616/jekyll-TeXt-theme#218
 */

$(function copyToClipboard() {
	if (!clipboardAttr.enable) {
		return;
	}

	function clearTooltip(e) {
		e.currentTarget.setAttribute('class', 'cpbtn');
		e.currentTarget.removeAttribute('aria-label');
	}

	function showTooltip(elem, msg) {
		elem.setAttribute('class', 'cpbtn tooltipped tooltipped-s');
		elem.setAttribute('aria-label', msg);
	}

	function fallbackMessage(action) {
		let tooltipMsg = "Error occurred!";
		if (clipboardAttr.failed && clipboardAttr.failed.length > 0) {
			tooltipMsg = clipboardAttr.failed;
		}
		return tooltipMsg;
	}

	$("div.highlighter-rouge").each(function (index, element) {
		element.firstChild.insertAdjacentHTML('beforebegin', '<button class="cpbtn" data-clipboard-snippet><i class="far fa-copy"></i></button>');
	});
	let clipboardSnippets = new ClipboardJS('[data-clipboard-snippet]', {
		target: function (trigger) {
			return trigger.nextElementSibling;
		}
	});
	clipboardSnippets.on('success', function (e) {
		e.clearSelection();
		let tooltipMsg = "Copied!";
		if (clipboardAttr.succeed && clipboardAttr.succeed.length > 0) {
			tooltipMsg = clipboardAttr.succeed;
		}
		showTooltip(e.trigger, tooltipMsg);
	});
	clipboardSnippets.on('error', function (e) {
		showTooltip(e.trigger, fallbackMessage(e.action));
	});

	let btns = document.querySelectorAll('.cpbtn');
	for (let i = 0; i < btns.length; i++) {
		btns[i].addEventListener('mouseleave', clearTooltip);
		btns[i].addEventListener('blur', clearTooltip);
		btns[i].addEventListener('mouseenter', function (e) {
			let tooltipMsg = "Copy to clipboard";
			if (clipboardAttr.tooltip && clipboardAttr.tooltip.length > 0) {
				tooltipMsg = clipboardAttr.tooltip;
			}
			showTooltip(e.currentTarget, tooltipMsg);
		}, false);
	}
});