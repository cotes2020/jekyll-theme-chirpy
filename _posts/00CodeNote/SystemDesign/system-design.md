
- [System Design](#system-design)
    - [example](#example)
    - [Design Instagram.](#design-instagram)
      - [High Level System Design](#high-level-system-design)
      - [Database Schema](#database-schema)
      - [Data Size Estimation](#data-size-estimation)
      - [Component Design](#component-design)
    - [Design Tik-Tok](#design-tik-tok)
    - [Design twitter](#design-twitter)
    - [Design Uber](#design-uber)
    - [Design What’s up](#design-whats-up)
    - [Discussion and designing LRU cache.](#discussion-and-designing-lru-cache)
    - [Design a garbage collection system.](#design-a-garbage-collection-system)
    - [Design a system to capture unique addresses in the entire world.](#design-a-system-to-capture-unique-addresses-in-the-entire-world)
    - [Design a recommendation system for products.](#design-a-recommendation-system-for-products)
    - [Design a toll system for highways.](#design-a-toll-system-for-highways)
    - [Design URL Shortener.](#design-url-shortener)
    - [Design Instant Messenger.](#design-instant-messenger)
    - [Design Elevator system.](#design-elevator-system)
    - [Design distributed caching system.](#design-distributed-caching-system)
    - [Design Amazon Locker Service.](#design-amazon-locker-service)
    - [Design Amazon Best Seller Item Service.](#design-amazon-best-seller-item-service)
    - [Design a URL shortening service.](#design-a-url-shortening-service)
    - [design a traffic control system](#design-a-traffic-control-system)
    - [design a search typeahead](#design-a-search-typeahead)
    - [designing a Web Crawler](#designing-a-web-crawler)
    - [design a warehouse system for Amazon.com](#design-a-warehouse-system-for-amazoncom)
    - [design `Amazon.com` so it can handle 10x more traffic than today](#design-amazoncom-so-it-can-handle-10x-more-traffic-than-today)
    - [design Amazon.com's database (customers, orders, products, etc.)](#design-amazoncoms-database-customers-orders-products-etc)
    - [Design a counters system for online services.](#design-a-counters-system-for-online-services)
    - [Design a game of chess.](#design-a-game-of-chess)
    - [Design a parking garage.](#design-a-parking-garage)
    - [Design an email sender that can send 100,000,000 emails.](#design-an-email-sender-that-can-send-100000000-emails)
    - [Design a video streaming service.](#design-a-video-streaming-service)
    - [Design an online bookstore.](#design-an-online-bookstore)
    - [Design a global file storage service or sharing service.](#design-a-global-file-storage-service-or-sharing-service)
    - [Design an API rate limiter.](#design-an-api-rate-limiter)
    - [Design a proximity server.](#design-a-proximity-server)
    - [Design a type-ahead service.](#design-a-type-ahead-service)
    - [How do you design a vending machine in Java?](#how-do-you-design-a-vending-machine-in-java)
    - [How to design a limit order book for trading systems?](#how-to-design-a-limit-order-book-for-trading-systems)
    - [How do you design an elevator system?](#how-do-you-design-an-elevator-system)
    - [go about designing an e-commerce website](#go-about-designing-an-e-commerce-website)
    - [go about designing the e-commerce website using microservices? How will you handle transactions](#go-about-designing-the-e-commerce-website-using-microservices-how-will-you-handle-transactions)
    - [Create an autocomplete feature like word suggestions on search engines. How will you scale it to millions of users?](#create-an-autocomplete-feature-like-word-suggestions-on-search-engines-how-will-you-scale-it-to-millions-of-users)



- ref
  - https://www.educative.io/courses/grokking-the-system-design-interview/m2yDVZnQ8lG
  - https://igotanoffer.com/blogs/tech/system-design-interviews
  -



---




# System Design



1. object
2. use amount
3. TPS: transaction per sec
4. additional service
   1. payment
5. use case
6. database, entity

dialog

key component
- Front end
  - user UI
  - invoke the search api
  - parse the parameter
  - search api call the book api
  - response page
- web server
  - micro service = API
  - service for book
  - service for api
  - service for payment
  - service for search
- API (domain/resource.parameter)
  - add a book
    - HTTP POST
      - xyz.com/books/
      - json:
      - {id:x, auther:x, price:x}
- database
  - identify key component
    - customer (name, id, address)
    - book (id, price, amount, version)
    - oerder (id, amount, date, paymentmethod)




### example

---


### Design Instagram.

focus on the following set of requirements while designing Instagram:

Functional Requirements
- Users should be able to upload/download/view photos.
- Users can perform searches based on photo/video titles.
- Users can follow other users.
- The system should generate and display a user’s News Feed consisting of top photos from all the people the user follows.

Non-functional Requirements
- Our service needs to be highly available.
- The acceptable latency of the system is 200ms for News Feed generation.
- Consistency can take a hit (in the interest of availability) if a user doesn’t see a photo for a while; it should be fine.
- The system should be highly reliable; any uploaded photo or video should never be lost.


Not in scope: Adding tags to photos, searching photos on tags, commenting on photos, tagging users to photos, who to follow, etc.


Design Considerations#
- The system would be read-heavy, so we will focus on building a system that can retrieve photos quickly.
- Practically, users can upload as many photos as they like; therefore, efficient management of storage should be a crucial factor in designing this system.
- Low latency is expected while viewing photos.
- Data should be 100% reliable. If a user uploads a photo, the system will guarantee that it will never be lost.


Capacity Estimation and Constraints#
- Let’s assume we have 500M total users, with 1M daily active users.
- 2M new photos every day, 23 new photos every second.
- Average photo file size => 200KB
- Total space required for 1 day of photos: 2M * 200KB => 400 GB
- Total space required for 10 years: 400GB * 365 (days a year) * 10 (years) ~= 1425TB


#### High Level System Design
- At a high-level, we need to support two scenarios, one to upload photos and the other to view/search photos.
- Our service would need some object storage servers to store photos and some database servers to store metadata information about the photos.


#### Database Schema
> 💡 Defining the DB schema in the early stages of the interview would help to understand the data flow among various components and later would guide towards data partitioning.
- We need to store data about users, their uploaded photos, and the people they follow.
- The Photo table will store all data related to a photo; we need to have an index on (PhotoID, CreationDate) since we need to fetch recent photos first.

![Screen Shot 2022-05-09 at 17.58.09](https://i.imgur.com/1bEh1oC.png)

- A straightforward approach for storing the above schema would be to use an RDBMS like MySQL since we require joins.
  - But relational databases come with their challenges, especially when we need to scale them.
  - SQL vs. NoSQL.

- We can store photos in a **distributed file storage** like HDFS or S3.

- We can store the above schema in a **distributed key-value store** to enjoy the benefits offered by NoSQL. All the metadata related to photos can go to a table where the ‘key’ would be the ‘PhotoID’ and the ‘value’ would be an object containing PhotoLocation, UserLocation, CreationTimestamp, etc.

- NoSQL stores
  - always maintain a certain number of replicas to offer reliability.
  - Also, in such data stores, deletes don’t get applied instantly;
  - data is retained for certain days (to support undeleting) before getting removed from the system permanently.


#### Data Size Estimation

- how much data will be going into each table and how much total storage we will need for 10 years.
- User: Assuming each “int” and “dateTime” is four bytes, each row in the User’s table will be of 68 bytes:
  - UserID (4 bytes) + Name (20 bytes) + Email (32 bytes) + DateOfBirth (4 bytes) + CreationDate (4 bytes) + LastLogin (4 bytes) = 68 bytes
  - If we have 500 million users, we will need 32GB of total storage.
  - 500 million * 68 ~= 32GB
- Photo: Each row in Photo’s table will be of 284 bytes:

  - PhotoID (4 bytes) + UserID (4 bytes) + PhotoPath (256 bytes) + PhotoLatitude (4 bytes) + PhotoLongitude(4 bytes) + UserLatitude (4 bytes) + UserLongitude (4 bytes) + CreationDate (4 bytes) = 284 bytes
  - If 2M new photos get uploaded every day, we will need 0.5GB of storage for one day:
  - 2M * 284 bytes ~= 0.5GB per day
  - For 10 years we will need 1.88TB of storage.
- UserFollow: Each row in the UserFollow table will consist of 8 bytes. If we have 500 million users and on average each user follows 500 users. We would need 1.82TB of storage for the UserFollow table:
  - 500 million users * 500 followers * 8 bytes ~= 1.82TB
  - Total space required for all tables for 10 years will be 3.7TB:
  - 32GB + 1.88TB + 1.82TB ~= 3.7TB


#### Component Design
> Separating photos’ read and write requests will also allow us to scale and optimize each of these operations independently.

Photo uploads (or writes) can be slow as they have to go to the disk, whereas reads will be faster, especially if they are being served from cache.

Uploading users can consume all the available connections, as uploading is a slow process. This means that ‘reads’ cannot be served if the system gets busy with all the ‘write’ requests.

- We should keep in mind that web servers have a connection limit before designing our system. If we assume that a web server can have a maximum of 500 connections at any time, then it can’t have more than 500 concurrent uploads or reads.
- To handle this bottleneck, we can split reads and writes into separate services.
- We will have dedicated servers for reads and different servers for writes to ensure that uploads don’t hog the system.



![Screen Shot 2022-05-09 at 18.03.53](https://i.imgur.com/eUsfWpd.png)

![Screen Shot 2022-05-09 at 18.04.05](https://i.imgur.com/qlMgYPB.png)





---


### Design Tik-Tok


---


### Design twitter


---


### Design Uber


---


### Design What’s up


---


### Discussion and designing LRU cache.


---


### Design a garbage collection system.


---


### Design a system to capture unique addresses in the entire world.


---


### Design a recommendation system for products.


---


### Design a toll system for highways.


---


### Design URL Shortener.


---


### Design Instant Messenger.


---


### Design Elevator system.


---


### Design distributed caching system.


---


### Design Amazon Locker Service.


---


### Design Amazon Best Seller Item Service.


---




---


### Design a URL shortening service.
While designing the URL shortening service, your ideal solution should:
- Create a unique URL ID while shortening a URL
- Handle redirects
- Delete expired URLs
- Have an upper limit of the number of characters for the shortened URL




### design a traffic control system

- Consider all phase transitions (From red to green, red to orange to green, and so on.)
- Be clear on the conditions in which a certain transition will take place
- Consider pedestrian crossing requirements
- Determine cycle length
- Determine clearance time
- Apportion green light time appropriately
- The traffic control system’s behavior will depend on the state of the traffic control system. Explain all your considerations when stating your solution and reasons for trade-offs made, if any.




### design a search typeahead

- Store previous search queries in the database
- Keep the data fresh
- Find the appropriate matches to the entered string
- Handle the queries per second - to be automatically handled by the system
- DIsplay the best matches from strings contained in the database


### designing a Web Crawler

- Prioritize web pages that are dynamic as these pages appear more frequently in search engine rankings
- The crawler should not be unbounded on the same domain
- Build a system that constantly tracks new web pages




---



### design a warehouse system for Amazon.com




---



### design `Amazon.com` so it can handle 10x more traffic than today




---



### design Amazon.com's database (customers, orders, products, etc.)




---



### Design a counters system for online services.




---



### Design a game of chess.




---



### Design a parking garage.




---



### Design an email sender that can send 100,000,000 emails.

You have five machines. How could you do it efficiently?




---



### Design a video streaming service.




---



### Design an online bookstore.




---



### Design a global file storage service or sharing service.




---



### Design an API rate limiter.




---



### Design a proximity server.




---



### Design a type-ahead service.




---



### How do you design a vending machine in Java?




---



### How to design a limit order book for trading systems?




---



### How do you design an elevator system?




---



### go about designing an e-commerce website




---



### go about designing the e-commerce website using microservices? How will you handle transactions


---


### Create an autocomplete feature like word suggestions on search engines. How will you scale it to millions of users?


















.
