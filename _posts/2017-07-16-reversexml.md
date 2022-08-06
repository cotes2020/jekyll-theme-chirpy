---
title:  "Reverse XML - lessons that I learned for millennial sake"
date:   2017-07-16 7:24 AM 
category: works 
tags: [tech]
layout: post
---

In April of 2001, I was so much into [XML](https://en.wikipedia.org/wiki/XML) and had some ideas of my own. I was working as a software engineer at [Knowledgeview](http://www.knowledgeview.com) between 1999 and 2001. The company was heavily focused on developing content syndication software for news agencies and newspapers. At the time, Perl language was still dominant for parsing text, and Java was the popular language for websites. Knowledgeview was adamant about using standard specifications, including NewsML, NITF, RSS, and more. XML, initially outlined in 1998 was starting to become a hot topic at the company. As a young programmer, working mainly from the Lebanon office, I had lesser exposure to the technologies that the London office was working on. That did not stop me from trying to innovate. On April 23, 2001, I sent an email to the XML mail distro at Knowledgeview that said:Date: Mon, 23 Apr 2001 11:39:03 +0100

"Attached is a 4-page white paper about a concept that crossed my mind last week. The idea initially started after a brief conversation with Dr. Ali about XML, in which he mentioned that not all companies might integrate XML in their applications. Such a remark made me think of solutions that would keep one form of framework for companies to exchange their data based on customized and different structuring without resorting to applications'modifications (expensive) but where standards (like any form of XML) still apply"

If the industry is heading towards XML as a standard form of communication across systems or applications, and if some companies may not be quick to jump onto XML, why not generate an orchestration mechanism between company A and company B to share data. The steps would be as follows:

  1. let each company declare its set of _set delimiters D_ for its content and publish the format on a common repository
  2. define a set of XSL rules that convert each set of _delimiters D_ from (1) into a universal _XML X_ format.
  3. anytime a company Y would like to leverage date from company X, company Y would query the common repository for  
    company X data specification and execute the set of rules in 2 to convert the text from Company X into the format  
    needed for Company Y.

I named the technique **Reverse XML**

I did not hear from anyone in the company about my idea. I was 27 at the time and was still young in the industry. I did not push myself or know any better ways to articulate my idea other than just emailing. A few months later, I left the company, not because of this but because I decided to move to the United States and build a new future with my wife.

Why am I saying all this? I thought I had a great idea at the same time. Given that I had limited resources, I did not know that there might have been a similar product out there. Maybe if I know at that time what know now I could have been more aggressive in marketing my idea. Moreover, even if I did not hear from my management, I should have tried another way. Nevertheless, I was proud of my idea and the name that I gave it **Reverse XML**. I tried to be imaginative and was thinking big. In my conclusion I wrote:

"The application may be established as a free service where the followin could be our revenue:

  1. Our database would include all companies' source formats, where our content-representation language that should handle all of these formats may create a data bridge between all those companies whose applications are not XML-friendly yet.
  2. Having said point 1, our database would also be valuable because we will be able to market-focus our products that may be of interest to these companies.

Note: "Reverse XML" may be free to use, and our company may provide as a paid service the option to write the client's content key files. 

I was thinking of open-sourcing the solution but provide a paid service for assisting companies.

Who knows, maybe it would have been a great business opportunity or a great success story. Perhaps this idea might have turned big, just like JSON format nowadays. What if I patented the idea or made something more of it? There is no shortage of one's tendency to dream and think of significant accomplishments. Why not Unfortunately, I did not push for it, and, at the same time, I could not convey its value in a better presentable fashion.

I later received a call from management, but eventually, the idea was not understood or accepted. I still believe that, at the time, this idea could have had great potential. But that does not matter. What matters is not to give up on your ideas. Push for them. That said, if you are in the late twenties, early thirties, or any age if you have a great idea, push for it with your heart and soul. It might win big, and if it does not, learn from your mistakes to do something even better the next time. Don't just wait for someone to call you … be proactive and make the call - not once but more.

You can download the letter that I wrote in 2001 [here](/assets/files/reverseXML.pdf).

