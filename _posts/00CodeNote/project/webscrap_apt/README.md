
- [webscrap\_apt](#webscrap_apt)
  - [Usage](#usage)


# webscrap_apt

## Usage
- Run `python ./_posts/00CodeNote/project/webscrap_apt/play.py -t all  `
- generate csv `apt_YYYYMMDD.csv` in `./apt_output`
- update `APT-xx.png`


```bash

# run for all apt
$ python ./_posts/00CodeNote/project/webscrap_apt/play.py -t all
INFO:__main__:======= Apt_scrapper loaded at Tue Mar 14 23:20:51 2023
INFO:__main__:============ Apt_scrapper run for url: {'talisman': 'https://www.livetalisman.com/redmond/talisman/conventional/', 'modera': 'https://www.moderaredmond.com/redmond/modera-redmond/conventional/'} ============

INFO:__main__:======= Target Apartment: talisman =======
INFO:__main__:======= Load info from https://www.livetalisman.com/redmond/talisman/conventional/ =======
INFO:__main__:======= Got info for Apartment: talisman =======

INFO:__main__:======= Target Apartment: modera =======
INFO:__main__:======= Load info from https://www.moderaredmond.com/redmond/modera-redmond/conventional/ =======
INFO:__main__:🟢:S01;Beds/Baths;Studio / 1 ba;Rent;Starting from $1,812/month;Deposit;$300;Sq.Ft;477+;Only One Left!;Details
INFO:__main__:🟢:S01L;Beds/Baths;Studio / 1 ba;Rent;Starting from $2,715/month;Deposit;$300;Sq.Ft;641+;Available Apr 10, 2023;Details
INFO:__main__:🟢:A03;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,071/month;Deposit;$300;Sq.Ft;553+;Only One Left!;Details
INFO:__main__:🟢:A05;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,025/month;Deposit;$300;Sq.Ft;596;Available Apr 02, 2023;Details
INFO:__main__:🟢:A05L;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,432/month;Deposit;$300;Sq.Ft;735;Available Apr 05, 2023;Details
INFO:__main__:🟢:A06;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,174/month;Deposit;$300;Sq.Ft;606;Only One Left!;Details
INFO:__main__:🟢:A07;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,462/month;Deposit;$300;Sq.Ft;634+;Available Mar 22, 2023;Details
INFO:__main__:🟢:A09;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,195/month;Deposit;$300;Sq.Ft;640;2 Available;Details
INFO:__main__:🟢:A10;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,383/month;Deposit;$300;Sq.Ft;672+;2 Available;Details
INFO:__main__:🟢:A12;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,706/month;Deposit;$300;Sq.Ft;751+;Available Mar 18, 2023;Details
INFO:__main__:🟢:A14D;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,421/month;Deposit;$300;Sq.Ft;777+;Only One Left!;Details
INFO:__main__:🟢:A16D;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,882/month;Deposit;$300;Sq.Ft;882;Available Mar 22, 2023;Details
INFO:__main__:🟢:B01;Beds/Baths;2 bd/2 ba;Rent;Starting from $3,561/month;Deposit;$300;Sq.Ft;961;Only One Left!;Details
INFO:__main__:🟢:B03;Beds/Baths;2 bd/2 ba;Rent;Starting from $3,592/month;Deposit;$300;Sq.Ft;1,039+;Only One Left!;Details
INFO:__main__:🟢:B07D;Beds/Baths;2 bd/2 ba;Rent;Starting from $3,645/month;Deposit;$300;Sq.Ft;1,233;Only One Left!;Details
INFO:__main__:🟢:S01L;Beds/Baths;Studio / 1 ba;Rent;Starting from $2,715/month;Deposit;$300;Sq.Ft;641+;Available Apr 10, 2023;Details
INFO:__main__:🟢:A05L;Beds/Baths;1 bd/1 ba;Rent;Starting from $2,432/month;Deposit;$300;Sq.Ft;735;Available Apr 05, 2023;Details
INFO:__main__:======= Got info for Apartment: modera =======

INFO:__main__:
======= creating file: apt_20230314.csv =======
INFO:__main__:======= filing file: apt_20230314.csv =======
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'S01', 'Beds/Baths': 'Studio\xa0/ 1 ba', 'Rent': 'Starting from $1,812/month', 'Deposit': '$300', 'Sq.Ft': '477+', 'Limited_Time_Offer': '/', 'Available': 'Only One Left!'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'S01L', 'Beds/Baths': 'Studio\xa0/ 1 ba', 'Rent': 'Starting from $2,715/month', 'Deposit': '$300', 'Sq.Ft': '641+', 'Limited_Time_Offer': '/', 'Available': 'Available Apr 10, 2023'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A03', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,071/month', 'Deposit': '$300', 'Sq.Ft': '553+', 'Limited_Time_Offer': '/', 'Available': 'Only One Left!'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A05', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,025/month', 'Deposit': '$300', 'Sq.Ft': '596', 'Limited_Time_Offer': '/', 'Available': 'Available Apr 02, 2023'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A05L', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,432/month', 'Deposit': '$300', 'Sq.Ft': '735', 'Limited_Time_Offer': '/', 'Available': 'Available Apr 05, 2023'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A06', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,174/month', 'Deposit': '$300', 'Sq.Ft': '606', 'Limited_Time_Offer': '/', 'Available': 'Only One Left!'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A07', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,462/month', 'Deposit': '$300', 'Sq.Ft': '634+', 'Limited_Time_Offer': '/', 'Available': 'Available Mar 22, 2023'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A09', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,195/month', 'Deposit': '$300', 'Sq.Ft': '640', 'Limited_Time_Offer': '/', 'Available': '2 Available'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A10', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,383/month', 'Deposit': '$300', 'Sq.Ft': '672+', 'Limited_Time_Offer': '/', 'Available': '2 Available'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A12', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,706/month', 'Deposit': '$300', 'Sq.Ft': '751+', 'Limited_Time_Offer': '/', 'Available': 'Available Mar 18, 2023'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A14D', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,421/month', 'Deposit': '$300', 'Sq.Ft': '777+', 'Limited_Time_Offer': '/', 'Available': 'Only One Left!'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A16D', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,882/month', 'Deposit': '$300', 'Sq.Ft': '882', 'Limited_Time_Offer': '/', 'Available': 'Available Mar 22, 2023'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'B01', 'Beds/Baths': '2 bd/2 ba', 'Rent': 'Starting from $3,561/month', 'Deposit': '$300', 'Sq.Ft': '961', 'Limited_Time_Offer': '/', 'Available': 'Only One Left!'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'B03', 'Beds/Baths': '2 bd/2 ba', 'Rent': 'Starting from $3,592/month', 'Deposit': '$300', 'Sq.Ft': '1,039+', 'Limited_Time_Offer': '/', 'Available': 'Only One Left!'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'B07D', 'Beds/Baths': '2 bd/2 ba', 'Rent': 'Starting from $3,645/month', 'Deposit': '$300', 'Sq.Ft': '1,233', 'Limited_Time_Offer': '/', 'Available': 'Only One Left!'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'S01L', 'Beds/Baths': 'Studio\xa0/ 1 ba', 'Rent': 'Starting from $2,715/month', 'Deposit': '$300', 'Sq.Ft': '641+', 'Limited_Time_Offer': '/', 'Available': 'Available Apr 10, 2023'}
INFO:__main__:{'Apt': 'modera', 'Floor_plan': 'A05L', 'Beds/Baths': '1 bd/1 ba', 'Rent': 'Starting from $2,432/month', 'Deposit': '$300', 'Sq.Ft': '735', 'Limited_Time_Offer': '/', 'Available': 'Available Apr 05, 2023'}
INFO:__main__:======= info loaded in the file apt_20230314.csv =======


$ python ./_posts/00CodeNote/project/webscrap_apt/img_play.py
INFO:__main__:
======= creating csv_files list: ['./_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230108.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230322.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230109.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230321.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230320.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230324.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230318.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230319.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230325.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230317.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230316.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230314.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230315.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230103.csv', './_posts/00CodeNote/project/webscrap_apt/apt_output/apt_20230104.csv'] =======
INFO:__main__:
======= update apt png for APT talisman =======
INFO:__main__:
======= update apt png for APT modera =======
```
