/**
 * Count pageviews form GA or local cache file.
 *
 * Dependences:
 *   - jQuery
 *   - countUp.js(https://github.com/inorganik/countUp.js)
 *
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2018-2019 Cotes Chung
 * MIT License
 */

function countUp(min, max, dest) {
  if (min < max) {
    var numAnim = new CountUp(dest, min, max);
    if (!numAnim.error) {
      numAnim.start();
    } else {
      console.error(numAnim.error);
    }
  }
}

function countPV(path, rows) {
  /* path permalink looks like: '/posts/post-title/' */
  var fileName = path.replace(/\/posts\//g, '').replace(/\//g, '.html'); /* e.g. post-title.html */
  var count = 0;

  var _v2_url = path.replace(/posts\//g, ''); /* the v2.0+ blog permalink: "/post-title/" */

  for (var i = 0; i < rows.length; ++i) {
    var gaPath = rows[i][0];
    if (gaPath == path ||
      gaPath == _v2_url ||
      gaPath.concat('/') == _v2_url ||
      gaPath.slice(gaPath.lastIndexOf('/') + 1) === fileName) { // old permalink record
      count += parseInt(rows[i][1]);
    }
  }

  return count;
}


function displayPageviews(data) {
  if (data === undefined) {
    return;
  }

  var hasInit = getInitStatus();
  var rows = data.rows;

  if ($("#post-list").length > 0) { // the Home page
    $(".post-preview").each(function() {
      var path = $(this).children("h1").children("a").attr("href");
      var count = countPV(path, rows);
      count = (count == 0 ? 1 : count);

      if (!hasInit) {
        $(this).find('.pageviews').text(count);
      } else {
        var initCount = parseInt($(this).find('.pageviews').text());
        if (count > initCount) {
          countUp(initCount, count, $(this).find('.pageviews').attr('id'));
        }
      }
    });

  } else if ($(".post").length > 0) { // the post
    var path = window.location.pathname;
    var count = countPV(path, rows);
    count = (count == 0 ? 1 : count);

    if (!hasInit) {
      $('#pv').text(count);
    } else {
      var initCount = parseInt($('#pv').text());
      if (count > initCount) {
        countUp(initCount, count, 'pv');
      }
    }
  }

}


var getInitStatus = (function() {
  var hasInit = false;
  return function() {
    if (hasInit) {
      return true;
    } else {
      hasInit = true;
      return false;
    }
  }
})();


$(function() {
  // load pageview if this page has .pageviews
  if ($('.pageviews').length > 0) {

    // Get data from daily cache.
    $.getJSON('/assets/data/pageviews.json', displayPageviews);

    $.getJSON('/assets/data/proxy.json', function(meta) {
      $.ajax({
        type: 'GET',
        url: meta.proxyUrl,
        dataType: 'jsonp',
        jsonpCallback: "displayPageviews",
        error: function(jqXHR, textStatus, errorThrown) {
          console.log("Failed to load pageviews from proxy server: " + errorThrown);
        }
      });

    });

  } // endif

});