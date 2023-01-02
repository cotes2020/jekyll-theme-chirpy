
// JavaScript DOM

// [link](https://www.youtube.com/watch?v=0ik6X4DJKCc&list=PLillGF-RfqbYE6Ik_EuXA2iZFcE082B3s)

// ## DOM

// DOM, user interface.
// - document object model
// - page is represented by the DOM
// - tree of nodes/elements created by the browser
// - javascript can be used to read/write/manipulate to the DOM
// - Object Oriented Representation


// 1
// EXAMINE THE DOCUMENT OBJECT //

// console.dir(document);  // all the info of the doc
// console.log(document.domain);
// console.log(document.URL);
// console.log(document.title);
// document.title =  123;
// console.log(document.title);
// console.log(document.doctype);
// console.log(document.head);
// console.log(document.body);
// console.log(document.all); // show the index under
// console.log(document.all[10]);
// document.all[10].textContent = 'Hello'; // dont use it to select, index will change
// console.log(document.forms);
// console.log(document.forms[0]);
// console.log(document.links);
// console.log(document.images); // dont have, empty HTMLcollection



// GetElementById //

// console.log(document.getElementById('header-title'));
// var headerTitle = document.getElementById('header-title');
// var header = document.getElementById('main-header');
// console.log(headerTitle);
// headerTitle.textContent = 'Hello';
// headerTitle.innerText = 'Goodbye';

// // three: textContent / innerText / innerHTML
// // textContent: shows style Item Lister 123
// // innerText: no style Item Lister
// // innerHTML: add <h3>

// console.log(headerTitle.textContent);
// console.log(headerTitle.innerText);
// headerTitle.innerHTML = '<h3>Hello</h3>';

// style change
// header.style.borderBottom = 'solid 3px #000';



// GetElementsByClassName //
// var items = document.getElementsByClassName('list-group-item');
// console.log(items);
// console.log(items[1]);
// items[1].textContent = 'Hello 2';
// items[1].style.fontWeight = 'bold';
// items[1].style.backgroundColor = 'yellow';

// // give style to all:
// // Gives error
// items.style.backgroundColor = '#f4f4f4';
// // solution: loop
// for(var i = 0; i < items.length; i++){
//   items[i].style.backgroundColor = '#f4f4f4';
// }



// GetElementsByTagName //
// var li = document.getElementsByTagName('li');
// console.log(li);
// console.log(li[1]);
// li[1].textContent = 'Hello 2';
// li[1].style.fontWeight = 'bold';
// li[1].style.backgroundColor = 'yellow';
// // give style to all:
// // Gives error
// //items.style.backgroundColor = '#f4f4f4';
// // solution: loop
// for(var i = 0; i < li.length; i++){
//   li[i].style.backgroundColor = '#f4f4f4';
// }




// querySelector //
// only grep the first one

// // #id
// var header = document.querySelector('#main-header');
// header.style.borderBottom = 'solid 4px #ccc';

// // only the firstone
// var input = document.querySelector('input');
// input.value = 'Hello World'

// // sepecify which one
// var submit = document.querySelector('input[type="submit"]');
// submit.value="SEND"


// var item = document.querySelector('.list-group-item');
// item.style.color = 'red';

// // select the last one
// var lastItem = document.querySelector('.list-group-item:last-child');
// lastItem.style.color = 'blue';

// // select the 2nd one
// var secondItem = document.querySelector('.list-group-item:nth-child(2)');
// secondItem.style.color = 'coral';




// querySelectorAll //

// // according to the class, id, tag any
// var titles = document.querySelectorAll('.title');
// console.log(titles);
// titles[0].textContent = 'Hello';

// // according to odd / even
// var odd = document.querySelectorAll('li:nth-child(odd)');
// var even= document.querySelectorAll('li:nth-child(even)');

// for(var i = 0; i < odd.length; i++){
//   odd[i].style.backgroundColor = '#f4f4f4';
//   even[i].style.backgroundColor = '#ccc';
// }



// 2
// TRAVERSING THE DOM //
// var itemList = document.querySelector('#items');
// // parentNode
// console.log(itemList.parentNode);
// itemList.parentNode.style.backgroundColor = '#f4f4f4';
// console.log(itemList.parentNode.parentNode); //container
// console.log(itemList.parentNode.parentNode.parentNode); //body
// // parentElement
// console.log(itemList.parentElement);
// itemList.parentElement.style.backgroundColor = '#f4f4f4';
// console.log(itemList.parentElement.parentElement.parentElement);

// // childNodes
// console.log(itemList.childNodes); // all

// // children
// console.log(itemList.children);   // item.recommand
// console.log(itemList.children[1]);
// itemList.children[1].style.backgroundColor = 'yellow';

// // FirstChild
// console.log(itemList.firstChild);        // all
// // firstElementChild
// console.log(itemList.firstElementChild); // item.recommand
// itemList.firstElementChild.textContent = 'Hello 1';

// // lastChild
// console.log(itemList.lastChild);
// // lastElementChild
// console.log(itemList.lastElementChild);
// itemList.lastElementChild.textContent = 'Hello 4';

// // nextSibling
// console.log(itemList.nextSibling);
// // nextElementSibling
// console.log(itemList.nextElementSibling);

// // previousSibling
// console.log(itemList.previousSibling);
// // previousElementSibling
// console.log(itemList.previousElementSibling);
// itemList.previousElementSibling.style.color = 'green';



// createElement

// Create a div
// var newDiv =  document.createElement('div');
// newDiv.className = 'hello';                 // Add class
// newDiv.id = 'hello1';                       // Add id
// newDiv.setAttribute('title', 'Hello Div');  // Add attr

// // Create text node
// var newDivText = document.createTextNode('Hello World');
// // Add text node to div
// newDiv.appendChild(newDivText);

// // Add the div to doc
// var up = document.querySelector('header .col-md-6');
// var h1 = document.querySelector('header h1');

// console.log(newDiv);

// newDiv.style.fontSize = '30px';

// up.insertBefore(newDiv, h1);





// 3
// EVENTS //

// // Mouse Event
// var button = document.getElementById('button').addEventListener('click', buttonClick);

// function buttonClick(e){
// //   console.log('Button clicked');
// //   document.getElementById('header-title').textContent = 'Changed';
// //   document.querySelector('#main').style.backgroundColor = '#f4f4f4';

// //   console.log(e);
// //   console.log(e.target); // the info of the element been fired
// //   console.log(e.target.id);
// //   console.log(e.target.className);
// //   console.log(e.target.classList);
// //   var output = document.getElementById('output');
// //   output.innerHTML = '<h3>'+e.target.id+'</h3>';

// //   console.log(e.type);  // what event: click

// //   console.log(e.clientX);  // position of the mouse from window
// //   console.log(e.clientY);

// //   console.log(e.offsetX);  // actual position of the mouse
// //   console.log(e.offsetY);

// //   console.log(e.altKey);  // if holding any key together
// //   console.log(e.ctrlKey);
// //   console.log(e.shiftKey);
// }


// function runEvent(e){
//     console.log('EVENT TYPE: '+e.type);
//     // output.innerHTML = '<h3>MouseX: ' + e.offsetX + '<h3>MouseY: ' + e.offsetY + '</h3>';   // track mouse with mouse move
//     // box.box.backgroundColor = "rgb("+e.offsetX+","+e.offsetY+",40)";  // change the color with the move

//     // document.getElementById('output').innerHTML = '<h3>'+e.target.value+'</h3>';
//     // document.body.style.display = 'none';
//     // console.log(e.target.value);

//     // e.preventDefault();
// }

// var button = document.getElementById('button');
// var box = document.getElementById('box');

// // different way to click
// button.addEventListener('click', runEvent);
// button.addEventListener('dblclick', runEvent);
// button.addEventListener('mousedown', runEvent);
// button.addEventListener('mouseup', runEvent);   // until release the mouse
// box.addEventListener('mouseenter', runEvent);   // when mouse enter
// box.addEventListener('mouseleave', runEvent);
// box.addEventListener('mouseover', runEvent); // both in and out
// box.addEventListener('mouseout', runEvent);  // check with mouseleave
// box.addEventListener('mousemove', runEvent);


// // Keyboard Event
// var itemInput = document.querySelector('input[type="text"]');
// var form = document.querySelector('form');
// var select = document.querySelector('select');

// itemInput.addEventListener('keydown', runEvent);
// itemInput.addEventListener('keyup', runEvent);
// itemInput.addEventListener('keypress', runEvent);

// itemInput.addEventListener('focus', runEvent);
// itemInput.addEventListener('blur', runEvent);

// itemInput.addEventListener('cut', runEvent);
// itemInput.addEventListener('paste', runEvent);

// itemInput.addEventListener('input', runEvent);

// // Select Events
// select.addEventListener('change', runEvent);
// select.addEventListener('input', runEvent);

// // Submit Events
// form.addEventListener('submit', runEvent);





// // // 4
// var form = document.getElementById('addForm');
// var itemList = document.getElementById('items');
// var filter = document.getElementById('filter');

// // Form submit event
// form.addEventListener('submit', addItem);
// // Delete event
// itemList.addEventListener('click', removeItem);
// // Filter event
// filter.addEventListener('keyup', filterItems);


// // Add item
// function addItem(e){
//     e.preventDefault();
//     // console.log(1);

//     // Create input as new li element
//     var li = document.createElement('li');
//     // Add class
//     li.className = 'list-group-item';
//     // Get input value
//     var newItem = document.getElementById('item').value;
//     // Add text node with input value
//     li.appendChild(document.createTextNode(newItem));

//     // Create del button element
//     var deleteBtn = document.createElement('button');
//     // Add classes to del button
//     deleteBtn.className = 'btn btn-danger btn-sm float-right delete';
//     // Append text node
//     deleteBtn.appendChild(document.createTextNode('X'));
//     // Append button to li
//     li.appendChild(deleteBtn);

//     // Append li to list
//     itemList.appendChild(li);
// }


// // Remove item
// function removeItem(e){
//   if(e.target.classList.contains('delete')){  // if click the class delete
//     if(confirm('Are You Sure?')){
//       var li = e.target.parentElement;
//       itemList.removeChild(li);
//     }
//   }
// }


// // Filter Items
// function filterItems(e){
//   // convert text to lowercase
//   var text = e.target.value.toLowerCase();
//   // Get lis, result is array
//   var items = itemList.getElementsByTagName('li');
//   // Convert to an array
//   Array.from(items).forEach(function(item){
//     var itemName = item.firstChild.textContent;
//     if(itemName.toLowerCase().indexOf(text) != -1){ // if true
//       item.style.display = 'block';
//     } else {
//       item.style.display = 'none';
//     }
//   });
// }
