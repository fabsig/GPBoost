$(document).ready(function() {

  $chunks = $('.fold');

  $chunks.each(function () {

    // add button to source code chunks
    if ( $(this).hasClass('s') ) {
      $('pre.r', this).prepend("<div class=\"showopt\">Show Code</div><br style=\"line-height:22px;\"/>");
      $('pre.r', this).children('code').attr('class', 'folded');
    }

    // add button to output chunks
    if ( $(this).hasClass('o') ) {
      $('pre:not(.r)', this).has('code').prepend("<div class=\"showopt\">Show Code</div><br style=\"line-height:22px;\"/>");
      $('pre:not(.r)', this).children('code:not(r)').addClass('folded');

      // add button to plots
      $(this).find('img').wrap('<pre class=\"plot\"></pre>');
      $('pre.plot', this).prepend("<div class=\"showopt\">Show Plot</div><br style=\"line-height:22px;\"/>");
      $('pre.plot', this).children('img').addClass('folded');

    }
  });

  // hide all chunks when document is loaded
  $('.folded').css('display', 'none')

  // function to toggle the visibility
  $('.showopt').click(function() {
    var label = $(this).html();
    if (label.indexOf("Show") >= 0) {
      $(this).html(label.replace("Show", "Hide"));
    } else {
      $(this).html(label.replace("Hide", "Show"));
    }
    $(this).siblings('code, img').slideToggle('fast', 'swing');
  });
});