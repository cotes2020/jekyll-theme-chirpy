/*
* This script make #search-result-wrap switch to hidden or shown automatically.
* © 2018-2019 Cotes Chung
* MIT License
*/

$(function() {

  var offset = 0;

  var btnCancel = $('#search-wrap + a');
  var btnSbTrigger = $('#sidebar-trigger');
  var btnSearchTrigger = $('#search-trigger');
  var btnCloseSearch = $('#search-wrap + a');
  var topbarTitle = $('#topbar-title');
  var searchWrap = $('#search-wrap');

  /*--- Actions in small screens ---*/

  btnSearchTrigger.click(function() {

    offset = $(window).scrollTop();

    $('body').addClass('no-scroll');
    // $('body').css('top', -offset + 'px');
    // $('html,body').addClass('input-focus');

    btnSbTrigger.addClass('hidden');
    topbarTitle.addClass('hidden');
    btnSearchTrigger.addClass('hidden');

    searchWrap.addClass('shown flex-grow-1');
    btnCancel.addClass('shown');

    $('#main').addClass('hidden');
    $('#search-result-wrap').addClass('shown');
    $('#search-input').focus();

  });

  btnCancel.click(function() {

    btnCancel.removeClass('shown');

    $('#search-input').val('');
    $('#search-results').empty();

    searchWrap.removeClass('shown flex-grow-1');

    btnSbTrigger.removeClass('hidden');
    topbarTitle.removeClass('hidden');
    btnSearchTrigger.removeClass('hidden');

    $('#main').removeClass('hidden');
    $('#search-result-wrap').removeClass('shown');

    $('body').removeClass('no-scroll');
    // $('html,body').removeClass('input-focus');

    $('html,body').scrollTop(offset);

  });

  /*--- Actions in large screens. ---*/

  var isShown = false;

  $('#search-input').on('input', function(){
    if (isShown == false) {
      offset = $(window).scrollTop();
      $('body,html').scrollTop(0);
      $('#search-result-wrap').addClass('shown');
      $('#main').addClass('hidden');
      isShown = true;
    }
  });

  $('#search-input').on('keyup', function(e){
    var input = $('#search-input').val();
    if (e.keyCode == 8 && input == '' && btnCloseSearch.css('display') == 'none') {
      $('#main').removeClass('hidden');
      $('#search-result-wrap').removeClass('shown');
      $('body,html').scrollTop(offset);
      isShown = false;
    }
  });

});
