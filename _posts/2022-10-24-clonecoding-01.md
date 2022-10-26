---
title: "react 클론코딩01"
author: cotes
categories: [study, react]
tag: [clonecoding]
math: true
mermaid: true
---

# 이현우 React(6)-Layout 연습문제

> 2022-10-21

## 완성본
![완성본](https://user-images.githubusercontent.com/105469077/197719637-d950b0f4-bf2f-4194-ab5f-602c17fa1f92.png)

## src

### index.js

```jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import Meta from './Meta';
import GlobalStyles from './GlobalStyles';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <Meta />
    <GlobalStyles />
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

### GlobalStyles.js

```jsx
import { createGlobalStyle } from "styled-components";
import reset from "styled-reset";

/**
 * 전역 스타일 시트를 정의한 객체
 * @type {GlobalStyleComponent<{}, DefaultTheme>}
 */

const GlobalStyles = createGlobalStyle`
  ${reset}

  * {
    font-family: 'Noto Sans KR';
  }

  a {
    text-decoration: none;
    color : #000;
  }

  ul {
    list-style: none;
  }

  body {
    margin: 0;
    padding: 0;
  }
`;

export default GlobalStyles;
```

### App.js

```jsx
import React from 'react';

import Navbar from './common/Navbar';
import Main from './pages/Main';
import Footer from './common/Footer';

const App = () => {
  return (
    <div className="App">
      <Navbar />
      <Main />
      <Footer />
    </div>
  );
}

export default App;
```

## common

### Navbar.js

```jsx
import React from 'react';
import styled from 'styled-components';

const NavbarComponent = styled.nav`
  background-color: white;
  box-shadow: 0 2px 5px 0 rgba(0,0,0,0.16),0 2px 10px 0 rgba(0,0,0,0.12);

  ul {
    display: flex;
    padding: 20px 15px;
    justify-content: space-between;
    gap: 1rem;
    letter-spacing : 2px;

    li:first-of-type {
      flex: 1 1 auto;
    }
  }

  
`

const navbar = () => {
  return (
    <NavbarComponent>
      <ul>
        <li><a href='#'>Gourmet au Catering</a></li>
        <li><a href=''>About</a></li>
        <li><a href=''>Menu</a></li>
        <li><a href=''>Contact</a></li>
      </ul>
    </NavbarComponent>
  )
}

export default navbar;
```

### Footer.js

```jsx
import React from 'react';
import styled from 'styled-components';

const FooterComponent = styled.div`
  height: 100px;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: rgba(0,0,0,0.05);

  a {
    text-decoration: underline;
  }
`

const Footer = () => {
  return (
    <FooterComponent>
      <p>Powered by <a href="#">w3.css</a></p>
    </FooterComponent>
  )
}

export default Footer
```

## pages/Main

### index.js

```jsx
import React from 'react';
import Jumbotron from './Jumbotron';
import About from './About';
import Menu from './Menu';
import Contact from './Contact';

const Main = () => {
  return (
    <div>
      <Jumbotron />
      <About />
      <Menu />
      <Contact />
    </div>
  )
}

export default Main;
```

### Jumbotron.js

```jsx
import React from 'react';
import styled from 'styled-components';
import img from '../../assets/img/hamburger.jpg';

const JumbotronComponent = styled.div`
  max-width: 1600px;
  margin: 0 auto;
  position: relative;

  h1 {
    position: absolute;
    bottom: 0;
    left: 0;
    padding: 20px;
    font-size: 48px;
  }

  img {
    width: 100%;
  }
`

const Jumbotron = () => {
  return (
    <JumbotronComponent>
      <img src= {img} alt='hambuger' />
      <h1>Le Catering</h1>
    </JumbotronComponent>
  )
}

export default Jumbotron;
```

### About.js

```jsx
import React from 'react';
import styled from 'styled-components';
import tableSetting from '../../assets/img/tablesetting2.jpg';

const AboutComponent = styled.div`
  display: flex;
  max-width: 1100px;
  margin: 4rem auto;
  padding-bottom: 4rem;
  gap: 40px;
  border-bottom: 1px solid rgba(0,0,0,0.1);
  flex-wrap: wrap;

  img {
    opacity: 0.7;
    width: 45%;
    min-width: 400px;
    flex: 1 1 auto;
  }

  div {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    width:45%;
    flex: 1 1 auto;

    h1 {
      font-size: 48px;
      letter-spacing : 2px;
    }

    p {

      &:first-of-type {
        margin-top: 2rem;
        font-size: 24px;
        letter-spacing : 2px;
      }

      &:not(:first-of-type) {
        font-size: 18px;
        letter-spacing: 1px;
        line-height: 26px;

        span {
          padding: 2px 4px;
          background-color: rgba(0,0,0,0.08);
        }
      }
    }
  }
`

const About = () => {
  return (
    <AboutComponent>
      <img src={tableSetting} alt="tableSetting" />
      <div>
        <h1>About Catering</h1>
        <p>Tradition since 1889</p>
        <p>The Catering was founded in blabla by Mr. Smith in lorem ipsum dolor sit amet, consectetur adipiscing elit consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute iruredolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.We only use <span>seasonal</span> ingredients.</p>
        <p>Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum consectetur adipiscing elit, sed do eiusmod temporincididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
      </div>
    </AboutComponent>
  )
}

export default About;
```

### Menu.js

```jsx
import React from 'react';
import styled from 'styled-components';

const MenuComponent = styled.div`
  display: flex;
  max-width: 1100px;
  margin: 4rem auto;
  gap: 2rem;
  justify-content: space-between;
  flex-wrap: wrap;
  padding-bottom: 4rem;
  border-bottom: 1px solid rgba(0,0,0,0.1);

  img {
    opacity: 0.7;
    flex: 1 1 auto;
  }

  div {
    display: flex;
    flex-direction: column;
    gap: 2rem;
    align-items: center;
    flex: 1 1 auto;

    h1 {
      font-size: 48px;
      letter-spacing: 2px;
    }

    ul {
      display: flex;
      flex-direction:column;
      flex: 1 1 auto;
      padding: 2rem 0;
      justify-content: space-around;

      p:first-of-type {
        font-size: 20px;
        letter-spacing: 6px;
      }

      p:not(:first-of-type) {
        margin-top: 1rem;
        color: gray;
      }
    }
  }

`

const Menu = () => {
  return (
    <MenuComponent>
      <div>
        <h1>Our Menu</h1>
        <ul>
          <li>
            <p>Bread Basket</p>
            <p>Assortment of fresh baked fruit breads and muffins 5.50</p>
          </li>
          <li>
            <p>Honey Almond Granola with Fruits</p>
            <p>Natural cereal of honey toasted oats, raisins, almonds and dates 7.00</p>
          </li>
          <li>
            <p>Belgian Waffle</p>
            <p>Vanilla flavored batter with malted flour 7.50</p>
          </li>
          <li>
            <p>Scrambled eggs</p>
            <p>Scrambled eggs, roasted red pepper and garlic, with green onions 7.50</p>
          </li>
          <li>
            <p>Blueberry Pancakes</p>
            <p>With syrup, butter and lots of berries 8.50</p>
          </li>
        </ul>
      </div>
      <img src="https://www.w3schools.com/w3images/tablesetting.jpg" alt="" />
    </MenuComponent>
  )
}

export default Menu
```

### Contact.js

```jsx
import React from 'react';
import styled from 'styled-components';

const ContactComponent = styled.div`
  max-width: 1100px;
  margin: 4rem auto;
  display: flex;
  flex-direction: column;
  gap: 2rem;

  h1 {
    font-size: 36px;
    letter-spacing: 4px;
  }

  p:nth-of-type(2) {
    font-size: 17px;
    color: #607d8b;
    font-weight: bold;
  }

  form {
    display: flex;
    flex-direction: column;
    gap: 2rem;

    input {
      padding: 1rem 0;
      border: none;
      border-bottom: 1px solid rgba(0,0,0,0.1);
      font-size: 18px;
    }

    button {
      align-self: flex-start;
      padding: 10px 15px;
      border: none;
    }
  }
`

const Contact = () => {
  return (
    <ContactComponent>
      <h1>Contact</h1>
      <p>We offer full-service catering for any event, large or small. We understand your needs and we will cater the food to satisfy the biggerst criteria of them all, both look and taste. Do not hesitate to contact us.</p>
      <p>Catering Service, 42nd Living St, 43043 New York, NY</p>
      <p>You can also contact us by phone 00553123-2323 or email catering@catering.com, or you can send us a message here:</p>
      <form>
        <input type="text" placeholder='Name'/>
        <input type="number" placeholder='How many people'/>
        <input type="date" />
        <input type="text" placeholder='Message \ Special requirements'/>
        <button>SEND MESSAGE</button>
      </form>
    </ContactComponent>
  )
}

export default Contact;
```

## 완성본

![완성본.png](%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A3%E1%86%BC%E1%84%89%E1%85%B5%E1%86%A8%2083610d3960924589b495a91492e78359/%25EC%2599%2584%25EC%2584%25B1%25EB%25B3%25B8.png)