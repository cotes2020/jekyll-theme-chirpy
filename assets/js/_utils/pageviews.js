/*
 * Count page views form GA or local cache file.
 *
 * Dependencies:
 *   - jQuery
 *   - countUp.js <https://github.com/inorganik/countUp.js>
 */

const getInitStatus = (function () {
  let hasInit = false;
  return () => {
    let ret = hasInit;
    if (!hasInit) {
      hasInit = true;
    }
    return ret;
  };
}());

const PvOpts = (function () {
  function hasContent(selector) {
    let content = $(selector).attr("content");
    return (typeof content !== "undefined" && content !== false);
  }

  return {
    getProxyEndpoint() {
      return $("meta[name=pv-proxy-endpoint]").attr("content");
    },
    getLocalData() {
      return $("meta[name=pv-cache-path]").attr("content");
    },
    hasProxyEndpoint() {
      return hasContent("meta[name=pv-proxy-endpoint]");
    },
    hasLocalData() {
      return hasContent("meta[name=pv-cache-path]");
    }
  }
}());

const PvStorage = (function () {
  const KEY_PV = "pv";
  const KEY_CREATION = "pv_created_date";

  function get(key) {
    return localStorage.getItem(key);
  }

  function set(key, val) {
    localStorage.setItem(key, val);
  }

  return {
    hasCache() {
      return (localStorage.getItem(KEY_PV) !== null);
    },
    getCache() {
      // get data from browser cache
      return JSON.parse(localStorage.getItem(KEY_PV));
    },
    saveCache(pv) {
      set(KEY_PV, pv);
      set(KEY_CREATION, new Date().toJSON());
    },
    isExpired() {
      let date = new Date(get(KEY_CREATION));
      date.setHours(date.getHours() + 1); // per hour
      return Date.now() >= date.getTime();
    },
    getAllPageviews() {
      return PvStorage.getCache().totalsForAllResults["ga:pageviews"];
    },
    newerThan(pv) {
      return PvStorage.getAllPageviews() > pv.totalsForAllResults["ga:pageviews"];
    },
    inspectKeys() {
      for(let i = 0; i < localStorage.length; i++){
        const key = localStorage.key(i);
        switch (key) {
          case KEY_PV:
          case KEY_CREATION:
            break;
          default:
            localStorage.clear();
            return;
        }
      }
    }
  };
}()); /* PvStorage */

function countUp(min, max, destId) {
  if (min < max) {
    let numAnim = new CountUp(destId, min, max);
    if (!numAnim.error) {
      numAnim.start();
    } else {
      console.error(numAnim.error);
    }
  }
}

function countPV(path, rows) {
  let count = 0;

  if (typeof rows !== "undefined" ) {
    for (let i = 0; i < rows.length; ++i) {
      const gaPath = rows[parseInt(i, 10)][0];
      if (gaPath === path) { /* path format see: site.permalink */
        count += parseInt(rows[parseInt(i, 10)][1], 10);
        break;
      }
    }
  }

  return count;
}

function tacklePV(rows, path, elem, hasInit) {
  let count = countPV(path, rows);
  count = (count === 0 ? 1 : count);

  if (!hasInit) {
    elem.text(new Intl.NumberFormat().format(count));
  } else {
    const initCount = parseInt(elem.text().replace(/,/g, ""), 10);
    if (count > initCount) {
      countUp(initCount, count, elem.attr("id"));
    }
  }
}

function displayPageviews(data) {
  if (typeof data === "undefined") {
    return;
  }

  let hasInit = getInitStatus();
  const rows = data.rows; /* could be undefined */

  if ($("#post-list").length > 0) { /* the Home page */
    $(".post-preview").each(function() {
      const path = $(this).find("a").attr("href");
      tacklePV(rows, path, $(this).find(".pageviews"), hasInit);
    });

  } else if ($(".post").length > 0) { /* the post */
    const path = window.location.pathname;
    tacklePV(rows, path, $("#pv"), hasInit);
  }
}

function fetchProxyPageviews() {
  if (PvOpts.hasProxyEndpoint()) {
    $.ajax({
      type: "GET",
      url: PvOpts.getProxyEndpoint(),
      dataType: "jsonp",
      jsonpCallback: "displayPageviews",
      success: (data, textStatus, jqXHR) => {
        PvStorage.saveCache(JSON.stringify(data));
      },
      error: (jqXHR, textStatus, errorThrown) => {
        console.log("Failed to load pageviews from proxy server: " + errorThrown);
      }
    });
  }
}

function loadPageviews(hasCache = false) {
  if (PvOpts.hasLocalData()) {
    fetch(PvOpts.getLocalData())
      .then((response) => response.json())
      .then((data) => {
        // The cache from the proxy will sometimes be more recent than the local one
        if (hasCache && PvStorage.newerThan(data)) {
          return;
        }
        displayPageviews(data);
        PvStorage.saveCache(JSON.stringify(data));
      })
      .then(() => {
        fetchProxyPageviews();
      });

  } else {
    fetchProxyPageviews();
  }

}

$(function() {
  if ($(".pageviews").length <= 0) {
    return;
  }

  PvStorage.inspectKeys();

  if (PvStorage.hasCache()) {
    displayPageviews(PvStorage.getCache());
    if (!PvStorage.isExpired()) {
      return;
    }
  }

  loadPageviews(PvStorage.hasCache());

});
