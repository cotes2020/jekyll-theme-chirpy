/**
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
})();

const PvOpts = (function () {
  function getContent(selector) {
    return $(selector).attr('content');
  }

  function hasContent(selector) {
    let content = getContent(selector);
    return typeof content !== 'undefined' && content !== false;
  }

  return {
    getProxyMeta() {
      return getContent('meta[name=pv-proxy-endpoint]');
    },
    getLocalMeta() {
      return getContent('meta[name=pv-cache-path]');
    },
    hasProxyMeta() {
      return hasContent('meta[name=pv-proxy-endpoint]');
    },
    hasLocalMeta() {
      return hasContent('meta[name=pv-cache-path]');
    }
  };
})();

const PvStorage = (function () {
  const Keys = {
    KEY_PV: 'pv',
    KEY_PV_SRC: 'pv_src',
    KEY_CREATION: 'pv_created_date'
  };

  const Source = {
    LOCAL: 'same-origin',
    PROXY: 'cors'
  };

  function get(key) {
    return localStorage.getItem(key);
  }

  function set(key, val) {
    localStorage.setItem(key, val);
  }

  function saveCache(pv, src) {
    set(Keys.KEY_PV, pv);
    set(Keys.KEY_PV_SRC, src);
    set(Keys.KEY_CREATION, new Date().toJSON());
  }

  return {
    keysCount() {
      return Object.keys(Keys).length;
    },
    hasCache() {
      return localStorage.getItem(Keys.KEY_PV) !== null;
    },
    getCache() {
      return JSON.parse(localStorage.getItem(Keys.KEY_PV));
    },
    saveLocalCache(pv) {
      saveCache(pv, Source.LOCAL);
    },
    saveProxyCache(pv) {
      saveCache(pv, Source.PROXY);
    },
    isExpired() {
      let date = new Date(get(Keys.KEY_CREATION));
      date.setHours(date.getHours() + 1); // per hour
      return Date.now() >= date.getTime();
    },
    isFromLocal() {
      return get(Keys.KEY_PV_SRC) === Source.LOCAL;
    },
    isFromProxy() {
      return get(Keys.KEY_PV_SRC) === Source.PROXY;
    },
    newerThan(pv) {
      return (
        PvStorage.getCache().totalsForAllResults['ga:pageviews'] >
        pv.totalsForAllResults['ga:pageviews']
      );
    },
    inspectKeys() {
      if (localStorage.length !== PvStorage.keysCount()) {
        localStorage.clear();
        return;
      }

      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        switch (key) {
          case Keys.KEY_PV:
          case Keys.KEY_PV_SRC:
          case Keys.KEY_CREATION:
            break;
          default:
            localStorage.clear();
            return;
        }
      }
    }
  };
})(); /* PvStorage */

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

  if (typeof rows !== 'undefined') {
    for (let i = 0; i < rows.length; ++i) {
      const gaPath = rows[parseInt(i, 10)][0];
      if (gaPath === path) {
        /* path format see: site.permalink */
        count += parseInt(rows[parseInt(i, 10)][1], 10);
        break;
      }
    }
  }

  return count;
}

function tacklePV(rows, path, elem, hasInit) {
  let count = countPV(path, rows);
  count = count === 0 ? 1 : count;

  if (!hasInit) {
    elem.text(new Intl.NumberFormat().format(count));
  } else {
    const initCount = parseInt(elem.text().replace(/,/g, ''), 10);
    if (count > initCount) {
      countUp(initCount, count, elem.attr('id'));
    }
  }
}

function displayPageviews(data) {
  if (typeof data === 'undefined') {
    return;
  }

  let hasInit = getInitStatus();
  const rows = data.rows; /* could be undefined */

  if ($('#post-list').length > 0) {
    /* the Home page */
    $('.post-preview').each(function () {
      const path = $(this).find('a').attr('href');
      tacklePV(rows, path, $(this).find('.pageviews'), hasInit);
    });
  } else if ($('.post').length > 0) {
    /* the post */
    const path = window.location.pathname;
    tacklePV(rows, path, $('#pv'), hasInit);
  }
}

function fetchProxyPageviews() {
  if (PvOpts.hasProxyMeta()) {
    $.ajax({
      type: 'GET',
      url: PvOpts.getProxyMeta(),
      dataType: 'jsonp',
      jsonpCallback: 'displayPageviews',
      success: (data) => {
        PvStorage.saveProxyCache(JSON.stringify(data));
      },
      error: (jqXHR, textStatus, errorThrown) => {
        console.log(
          'Failed to load pageviews from proxy server: ' + errorThrown
        );
      }
    });
  }
}

function fetchLocalPageviews(hasCache = false) {
  return fetch(PvOpts.getLocalMeta())
    .then((response) => response.json())
    .then((data) => {
      if (hasCache) {
        // The cache from the proxy will sometimes be more recent than the local one
        if (PvStorage.isFromProxy() && PvStorage.newerThan(data)) {
          return;
        }
      }
      displayPageviews(data);
      PvStorage.saveLocalCache(JSON.stringify(data));
    });
}

export function initPageviews() {
  if ($('.pageviews').length <= 0) {
    return;
  }

  PvStorage.inspectKeys();

  if (PvStorage.hasCache()) {
    displayPageviews(PvStorage.getCache());

    if (PvStorage.isExpired()) {
      if (PvOpts.hasLocalMeta()) {
        fetchLocalPageviews(true).then(fetchProxyPageviews);
      } else {
        fetchProxyPageviews();
      }
    } else {
      if (PvStorage.isFromLocal()) {
        fetchProxyPageviews();
      }
    }
  } else {
    // no cached

    if (PvOpts.hasLocalMeta()) {
      fetchLocalPageviews().then(fetchProxyPageviews);
    } else {
      fetchProxyPageviews();
    }
  }
}
