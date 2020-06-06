/**
 * Count pageviews form GA or local cache file.
 *
 * Dependences:
 *   - jQuery
 *   - countUp.js <https://github.com/inorganik/countUp.js>
 *
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2018-2019 Cotes Chung
 * MIT License
 */

function countUp(min, max, destId) {
  if (min < max) {
    var numAnim = new CountUp(destId, min, max);
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
      gaPath.slice(gaPath.lastIndexOf('/') + 1) === fileName) { /* old permalink record */
      count += parseInt(rows[i][1]);
    }
  }

  return count;
}


function tacklePV(rows, path, elem, hasInit) {
  var count = countPV(path, rows);
  count = (count == 0 ? 1 : count);

  if (!hasInit) {
    elem.text(new Intl.NumberFormat().format(count));
  } else {
    var initCount = parseInt(elem.text().replace(/,/g, ''));
    if (count > initCount) {
      countUp(initCount, count, elem.attr('id'));
    }
  }
}


function displayPageviews(data) {
  if (data === undefined) {
    return;
  }

  var hasInit = getInitStatus();
  var rows = data.rows;

  if ($("#post-list").length > 0) { /* the Home page */
    $(".post-preview").each(function() {
      var path = $(this).children("div").children("h1").children("a").attr("href");
      tacklePV(rows, path, $(this).find('.pageviews'), hasInit);
    });

  } else if ($(".post").length > 0) { /* the post */
    var path = window.location.pathname;
    tacklePV(rows, path, $('#pv'), hasInit);
  }
}


var getInitStatus = (function() {
  var hasInit = false;
  return function() {
    let ret = hasInit;
    if (!hasInit) {
      hasInit = true;
    }
    return ret;
  }
})();


var PvCache = (function() {
  const KEY_PV = "pv";
  const KEY_CREATION = "pv-created-date";
  const KEY_PV_TYPE = "pv-type";

  var PvType = {
    ORIGIN: "origin",
    PROXY: "proxy"
  };

  function get(key) {
    return localStorage.getItem(key);
  }

  function set(key, val) {
    localStorage.setItem(key, val);
  }

  return {
    getData: function() {
      return JSON.parse(localStorage.getItem(KEY_PV) );
    },
    saveOriginCache: function(pv) {
      set(KEY_PV, pv);
      set(KEY_PV_TYPE, PvType.ORIGIN );
      set(KEY_CREATION, new Date().toJSON() );
    },
    saveProxyCache: function(pv) {
      set(KEY_PV, pv);
      set(KEY_PV_TYPE, PvType.PROXY );
      set(KEY_CREATION, new Date().toJSON() );
    },
    isOriginCache: function() {
      return get(KEY_PV_TYPE) == PvType.ORIGIN;
    },
    isProxyCache: function() {
      return get(KEY_PV_TYPE) == PvType.PROXY;
    },
    isExpired: function() {
      if (PvCache.isOriginCache() ) {
        let date = new Date(get(KEY_CREATION));
        date.setDate(date.getDate() + 1); /* fetch origin-data every day */
        return Date.now() >= date.getTime();

      } else if (PvCache.isProxyCache() ) {
        let date = new Date(get(KEY_CREATION) );
        date.setHours(date.getHours() + 1); /* proxy-data is updated every hour */
        return Date.now() >= date.getTime();
      }
      return false;
    },
    getAllPagevies: function() {
      return PvCache.getData().totalsForAllResults["ga:pageviews"];
    },
    newerThan: function(pv) {
      return PvCache.getAllPagevies() > pv.totalsForAllResults["ga:pageviews"];
    }
  };

})(); /* PvCache */


function fetchOriginPageviews(pvData) {
  if (pvData === undefined) {
    return;
  }
  displayPageviews(pvData);
  PvCache.saveOriginCache(JSON.stringify(pvData));
}


function fetchProxyPageviews() {
  let proxy = JSON.parse(proxyData); /* see file '/assets/data/pv-data.json' */
  $.ajax({
    type: 'GET',
    url: proxy.url,
    dataType: 'jsonp',
    jsonpCallback: "displayPageviews",
    success: function(data, textStatus, jqXHR) {
      PvCache.saveProxyCache(JSON.stringify(data));
    },
    error: function(jqXHR, textStatus, errorThrown) {
      console.log("Failed to load pageviews from proxy server: " + errorThrown);
    }
  });
}


$(function() {

  if ($('.pageviews').length > 0) {

    let cache = PvCache.getData();

    if (cache) {
      if (PvCache.isExpired()) {
        if (PvCache.isProxyCache() ) {
          let originPvData = pageviews ? JSON.parse(pageviews) : undefined;
          if (originPvData) {
            if (PvCache.newerThan(originPvData)) {
              displayPageviews(cache);
            } else {
              fetchOriginPageviews(originPvData);
            }
          }

          fetchProxyPageviews();

        } else if (PvCache.isOriginCache() ) {
          fetchOriginPageviews(originPvData);
          fetchProxyPageviews();
        }

      } else { /* still valid */
        displayPageviews(cache);

        if (PvCache.isOriginCache() ) {
          fetchProxyPageviews();
        }

      }

    } else {
      let originPvData = pageviews ? JSON.parse(pageviews) : undefined;
      fetchOriginPageviews(originPvData);
      fetchProxyPageviews();
    }

  }

});
