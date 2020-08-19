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

var getInitStatus = (function () {
  var hasInit = false;
  return () => {
    let ret = hasInit;
    if (!hasInit) {
      hasInit = true;
    }
    return ret;
  };
}());


var PvCache = (function () {
  const KEY_PV = "pv";
  const KEY_CREATION = "pv_created_date";
  const KEY_PV_SRC = "pv_source";

  var Source = {
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
    getData() {
      return JSON.parse(localStorage.getItem(KEY_PV) );
    },
    saveOriginCache(pv) {
      set(KEY_PV, pv);
      set(KEY_PV_SRC, Source.ORIGIN );
      set(KEY_CREATION, new Date().toJSON() );
    },
    saveProxyCache(pv) {
      set(KEY_PV, pv);
      set(KEY_PV_SRC, Source.PROXY );
      set(KEY_CREATION, new Date().toJSON() );
    },
    isFromOrigin() {
      return get(KEY_PV_SRC) === Source.ORIGIN;
    },
    isFromProxy() {
      return get(KEY_PV_SRC) === Source.PROXY;
    },
    isExpired() {
      if (PvCache.isFromOrigin() ) {
        let date = new Date(get(KEY_CREATION));
        date.setDate(date.getDate() + 1); /* update origin records every day */
        return Date.now() >= date.getTime();

      } else if (PvCache.isFromProxy() ) {
        let date = new Date(get(KEY_CREATION) );
        date.setHours(date.getHours() + 1); /* update proxy records per hour */
        return Date.now() >= date.getTime();
      }
      return false;
    },
    getAllPagevies() {
      return PvCache.getData().totalsForAllResults["ga:pageviews"];
    },
    newerThan(pv) {
      return PvCache.getAllPagevies() > pv.totalsForAllResults["ga:pageviews"];
    },
    inspectKeys() {
      if (localStorage.getItem(KEY_PV) === null
        || localStorage.getItem(KEY_PV_SRC) === null
        || localStorage.getItem(KEY_CREATION) === null) {
        localStorage.clear();
      }
    }
  };

}()); /* PvCache */

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
  var count = 0;

  if (typeof rows !== "undefined" ) {
    for (var i = 0; i < rows.length; ++i) {
      var gaPath = rows[parseInt(i, 10)][0];
      if (gaPath === path) { /* path format see: site.permalink */
        count += parseInt(rows[parseInt(i, 10)][1], 10);
        break;
      }
    }
  }

  return count;
}


function tacklePV(rows, path, elem, hasInit) {
  var count = countPV(path, rows);
  count = (count === 0 ? 1 : count);

  if (!hasInit) {
    elem.text(new Intl.NumberFormat().format(count));
  } else {
    var initCount = parseInt(elem.text().replace(/,/g, ""), 10);
    if (count > initCount) {
      countUp(initCount, count, elem.attr("id"));
    }
  }
}


function displayPageviews(data) {
  if (typeof data === "undefined") {
    return;
  }

  var hasInit = getInitStatus();
  var rows = data.rows; /* could be undefined */

  if ($("#post-list").length > 0) { /* the Home page */
    $(".post-preview").each(function() {
      var path = $(this).children("div").children("h1").children("a").attr("href");
      tacklePV(rows, path, $(this).find(".pageviews"), hasInit);
    });

  } else if ($(".post").length > 0) { /* the post */
    var path = window.location.pathname;
    tacklePV(rows, path, $("#pv"), hasInit);
  }
}


function fetchProxyPageviews() {
  $.ajax({
    type: "GET",
    url: proxyEndpoint, /* see: /assets/js/_pv-config.js */
    dataType: "jsonp",
    jsonpCallback: "displayPageviews",
    success: (data, textStatus, jqXHR) => {
      PvCache.saveProxyCache(JSON.stringify(data));
    },
    error: (jqXHR, textStatus, errorThrown) => {
      console.log("Failed to load pageviews from proxy server: " + errorThrown);
    }
  });
}


function fetchPageviews(fetchOrigin = true, filterOrigin = false) {
  /* pvCacheEnabled › see: /assets/js/_pv-config.js */
  if (pvCacheEnabled && fetchOrigin) {
    fetch("/assets/js/data/pageviews.json")
      .then((response) => response.json())
      .then((data) => {
        if (filterOrigin) {
          if (PvCache.newerThan(data)) {
            return;
          }
        }
        displayPageviews(data);
        PvCache.saveOriginCache(JSON.stringify(data));
      })
      .then(() => fetchProxyPageviews());

  } else {
    fetchProxyPageviews();
  }

}


$(function() {

  if ($(".pageviews").length > 0) {

    PvCache.inspectKeys();
    let cache = PvCache.getData();

    if (cache) {
      displayPageviews(cache);

      if (PvCache.isExpired()) {
        fetchPageviews(true, PvCache.isFromProxy());

      } else {

        if (PvCache.isFromOrigin()) {
          fetchPageviews(false);
        }

      }

    } else {
      fetchPageviews();
    }

  }

});
