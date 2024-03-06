
# Forms

[toc]

---

## forms
- forms are used to pass data
- different input element
- forms must have server with them

web development
- front end
  - what happens on the browser
  - HTML, CSS, Javascript
- back end
  - what server is handling
  - python, ruby, php, perl, java

**HTML Form Tags**:
- `<form>`
- `<label>`
- `<input>`


## Form elements attributes
`type`
- textfield
- text
- email
- password (mask)
- radiobutton (check only one)
- checkbox
- submit
- number (up and down)
- range (拉线)
- color
- date
- url

`name`
- almost all input types should have a name attribute
- the name attribute is assigned whatever value is input

`id`
- used for labels
- used for by javascript

`value`
- button: text inside the button.
- textfield: provide a default value.
- placeholder: like value, but will disapear.


---


## Validate

- the type
- the format
- the value


## how to validate
1. HTML5 `input types`
    - email, number, url
    - the browser validate the format of the input
    - when supported, it will halt the submit process for non-valid input.
    - if not supported, input just text.

2. HTML5 `attributes`
    - required, placeholder, min, max
    - halt the submit process if any required elements are empty.
    - alot of required, paired error novalidate.
    - `pattern`
      - work with input type = text and required the input have a specific form: (Regularexpression)
        - [0-9]{5}
        - [a-zA-Z]+
    - limit number: min, max...

```js
zip code with parttern:
<input type="text" name="zip-codeP" pattern="[0-9]{5}" required>

zip code with number:
<input type="number" name="zip-codeN" min="00000" max="99999" required>
// but 444 will go though too.
```

3. Javascript functions
    - write custom code to validate


---

## checkbox
- only multi choice: options share a single name.

```html
<label><input typr='checkbox' name='food' value='BigBlueberry'>BigBlueberry</label>
<label><input typr='checkbox' name='food' value='pizza'>pizza</label>
<label><input typr='checkbox' name='food' value='kale'>kale</label>
<input type='submit'>
<!-- label allow to push value to check it -->
```

## radio button

`name='food'`
- only one choice:
- 多选一的 options share a single name.

`checked='ture'`:
- 默认check项


```html
<label><input typr='radio' name='food' value='BigBlueberry' checked='ture'>BigBlueberry</label> <!-- 默认check项 -->
<label><input typr='radio' name='food' value='pizza'>pizza</label>
<label><input typr='radio' name='food' value='kale'>kale</label>

<label><input typr='radio' name='gender' value='female'>female</label>
<label><input typr='radio' name='gender' value='male'>male</label>

<input type='submit'>
```
