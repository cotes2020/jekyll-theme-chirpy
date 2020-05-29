/*!
  disqusLoader.js v1.0
  A JavaScript plugin for lazy-loading Disqus comments widget.
  -
  By Osvaldas Valutis, www.osvaldas.info
  Available for use under the MIT License
*/
(function(e,g,h,d){var a=e(g),k=function(o,n){var q,p;return function(){var t=this,s=arguments,r=+new Date;q&&r<q+o?(clearTimeout(p),p=setTimeout(function(){q=r,n.apply(t,s)},o)):(q=r,n.apply(t,s))}},m=false,j=false,i=false,c=false,f="unloaded",b=e(),l=function(){if(!b.length||b.data("disqusLoaderStatus")=="loaded"){return true}var n=a.scrollTop();if(b.offset().top-n>a.height()*j||n-b.offset().top-b.outerHeight()-(a.height()*j)>0){return true}e("#disqus_thread").removeAttr("id");b.attr("id","disqus_thread").data("disqusLoaderStatus","loaded");if(f=="loaded"){DISQUS.reset({reload:true,config:i})}else{g.disqus_config=i;if(f=="unloaded"){f="loading";e.ajax({url:c,async:true,cache:true,dataType:"script",success:function(){f="loaded"}})}}};a.on("scroll resize",k(m,l));e.disqusLoader=function(o,n){n=e.extend({},{laziness:1,throttle:250,scriptUrl:false,disqusConfig:false},n);j=n.laziness+1;m=n.throttle;i=n.disqusConfig;c=c===false?n.scriptUrl:c;b=(typeof o=="string"?e(o):o).eq(0);b.data("disqusLoaderStatus","unloaded");l()}})(jQuery,window,document);