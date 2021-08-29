/*
 * Copy code in code blocks to clipboard.
 * Porting from: kitian616/jekyll-TeXt-theme#218
 */

$(function copyToClipboard() {
	if (!enableCopyToClipboard) {
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
		var actionMsg = '';
		var actionKey = (action === 'cut' ? 'X' : 'C');
		if (/iPhone|iPad/i.test(navigator.userAgent)) {
			actionMsg = 'No support :(';
		} else if (/Mac/i.test(navigator.userAgent)) {
			actionMsg = 'Press ⌘-' + actionKey + ' to ' + action;
		} else {
			actionMsg = 'Press Ctrl-' + actionKey + ' to ' + action;
		}
		return actionMsg;
	}

	$("div.highlighter-rouge").each(function (index, element) {
		element.firstChild.insertAdjacentHTML('beforebegin', '<button class="cpbtn" data-clipboard-snippet><i class="far fa-copy"></i></button>');
	});
	var clipboardSnippets = new ClipboardJS('[data-clipboard-snippet]', {
		target: function (trigger) {
			return trigger.nextElementSibling;
		}
	});
	clipboardSnippets.on('success', function (e) {
		e.clearSelection();
		showTooltip(e.trigger, 'Copied!');
	});
	clipboardSnippets.on('error', function (e) {
		showTooltip(e.trigger, fallbackMessage(e.action));
	});

	var btns = document.querySelectorAll('.cpbtn');
	for (var i = 0; i < btns.length; i++) {
		btns[i].addEventListener('mouseleave', clearTooltip);
		btns[i].addEventListener('blur', clearTooltip);
		btns[i].addEventListener('mouseenter', function (e) {
			showTooltip(e.currentTarget, "copy to clipboard");
		}, false);
	}
});