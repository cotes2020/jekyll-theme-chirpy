---
title: 'xUnit - Getting Started'
date: 2019-01-27T21:57:57+01:00
author: Wolfgang Ofner
categories: [Programming, DevOps]
tags: ['C#', TDD, xUnit]
---
In this post, I will explain the basics of xUnit and how to write unit tests with it. xUnit is an open source testing framework for the .Net framework and was written by the inventor of NUnit v2. More details can be found on <a href="https://xunit.github.io/" target="_blank" rel="noopener">xUnit&#8217;s Github page</a>. xUnit is used by .Net core as the default testing framework and its major advantage over NUnit is that every test runs in isolation, which makes it impossible that test influence each other.

## My Setup

For writing unit tests I use the following NuGet packages and extensions:

  * xUnit for unit testing
  * <a href="http://xbehave.github.io/" target="_blank" rel="noopener">xBehave</a> for acceptance tests (xBehave is based on xUnit)
  * <a href="https://fluentassertions.com/" target="_blank" rel="noopener">FluentAssertions</a> for more readable assertions
  * <a href="https://fakeiteasy.github.io/" target="_blank" rel="noopener">FakeItEasy</a> to create fake objects
  * <a href="https://resharper-plugins.jetbrains.com/packages/xunitcontrib/" target="_blank" rel="noopener">xUnit Resharper Extension</a> for xUnit shortcuts in Visual Studio

The code for today&#8217;s demo can be found on <a href="https://github.com/WolfgangOfner/xUnit-Getting-Started" target="_blank" rel="noopener">Github</a>. Keep in mind that the tests are only for the demonstration of xUnit. The tests are barely useful.

## Execute a test with xUnit

For each class I want to test, I create a separate class to which I add tests in the name, for example, if I want to test the Employee class I name my test class EmployeeTests. To create a test method, you only have to add the Fact attribute to the method.

<div id="attachment_1531" style="width: 424px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Using-the-Fact-attritbue.jpg"><img aria-describedby="caption-attachment-1531" loading="lazy" class="wp-image-1531 size-full" src="/assets/img/posts/2019/01/Using-the-Fact-attritbue.jpg" alt="Using the xUnit Fact attribute" width="414" height="138" /></a>
  
  <p id="caption-attachment-1531" class="wp-caption-text">
    Using the Fact attribute
  </p>
</div>

That&#8217;s all. You can run the test and if the constructor of your Employee class sets the salary to 1000, the test will pass. I like to name the object I want to test testee. Another common name is sut which stands for system under test.

## Reducing code duplication

In the intro, I mentioned that every test runs in isolation in xUnit. This is done by creating a new instance for each test. Therefore the constructor is called for each test and can be used to initialize objects, which are needed for the tests. Since I will need the object of the Employee class in all my tests, I can initialize it in the constructor and don&#8217;t have to write the same code over and over in every test.

<div id="attachment_1532" style="width: 327px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Initialize-the-testee-in-the-constructor.jpg"><img aria-describedby="caption-attachment-1532" loading="lazy" class="size-full wp-image-1532" src="/assets/img/posts/2019/01/Initialize-the-testee-in-the-constructor.jpg" alt="Initialize the testee in the constructor" width="317" height="138" /></a>
  
  <p id="caption-attachment-1532" class="wp-caption-text">
    Initialize the testee in the constructor
  </p>
</div>

## Cleaning up after tests

Sometimes you have to do some cleanup like a database rollback or deleting a file after the tests were executed. Like the constructor, this can be done in a central place for all tests. To do that implement the IDisposable interface and implement the Dispose method. This method is called every time a test is finished.

<div id="attachment_1533" style="width: 348px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Implement-the-IDisposable-interface.jpg"><img aria-describedby="caption-attachment-1533" loading="lazy" class="size-full wp-image-1533" src="/assets/img/posts/2019/01/Implement-the-IDisposable-interface.jpg" alt="Implement the IDisposable interface" width="338" height="262" /></a>
  
  <p id="caption-attachment-1533" class="wp-caption-text">
    Implement the IDisposable interface
  </p>
</div>

## Executing tests several times with different parameters

Often you want to execute a test with different parameters, for example, if a valid age for your employee has to be between at least 18 and maximum 65 years, you want to test the edge cases (17, 18, 65, 66). Additionally, you might test negative numbers. You could write several asserts but this would be a lot of typing and not really practical. The solution for this is the Theory attribute in xUnit. A Theory allows you to pass values from different sources as parameters to your test method. With the InlineData attribute, you can add values for the parameter.

<div id="attachment_1534" style="width: 496px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Executing-the-same-method-with-several-input-variables.jpg"><img aria-describedby="caption-attachment-1534" loading="lazy" class="size-full wp-image-1534" src="/assets/img/posts/2019/01/Executing-the-same-method-with-several-input-variables.jpg" alt="Executing the same method with several input variables" width="486" height="215" /></a>
  
  <p id="caption-attachment-1534" class="wp-caption-text">
    Executing the same method with several input variables
  </p>
</div>

If you run this test method, five test cases will be executed.

## Skipping a test

Sometimes you don&#8217;t want a test to be executed. To ignore tests, add the Skip attribute and provide an info message.

<div id="attachment_1535" style="width: 283px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Skipping-a-test.jpg"><img aria-describedby="caption-attachment-1535" loading="lazy" class="size-full wp-image-1535" src="/assets/img/posts/2019/01/Skipping-a-test.jpg" alt="Skipping a test" width="273" height="100" /></a>
  
  <p id="caption-attachment-1535" class="wp-caption-text">
    Skipping a test
  </p>
</div>

## Grouping tests together

I barely use this feature but sometimes you want to group certain tests together. This can be for example all tests from one class and only some tests from another class. To do that use the Trait attribute and provide a name and category. You can apply this attribute to a class or to a single test.

<div id="attachment_1536" style="width: 221px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-method.jpg"><img aria-describedby="caption-attachment-1536" loading="lazy" class="size-full wp-image-1536" src="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-method.jpg" alt="Applying the Trait attribute to a test method" width="211" height="122" /></a>
  
  <p id="caption-attachment-1536" class="wp-caption-text">
    Applying the Trait attribute to a test method
  </p>
</div>

<div id="attachment_1537" style="width: 262px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-class.jpg"><img aria-describedby="caption-attachment-1537" loading="lazy" class="size-full wp-image-1537" src="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-class.jpg" alt="Applying the Trait attribute to a test class" width="252" height="289" /></a>
  
  <p id="caption-attachment-1537" class="wp-caption-text">
    Applying the Trait attribute to a test class
  </p>
</div>

If you run the tests and group the output by category, all traits with the same category will be grouped together.

<div id="attachment_1538" style="width: 290px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Grouped-test-output.jpg"><img aria-describedby="caption-attachment-1538" loading="lazy" class="size-full wp-image-1538" src="/assets/img/posts/2019/01/Grouped-test-output.jpg" alt="Grouped test output" width="280" height="154" /></a>
  
  <p id="caption-attachment-1538" class="wp-caption-text">
    Grouped test output
  </p>
</div>

## Add information to the test result output

By default, no output is generated when a test finished. For reporting reasons, it can be useful to add some information on what the test did to the output of the test. This can be done with the ITestOutputHelper. Pass it as parameter in the constructor of your test class and initialize a private field with it.

<div id="attachment_1539" style="width: 381px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Using-the-ITestOutputHelper.jpg"><img aria-describedby="caption-attachment-1539" loading="lazy" class="size-full wp-image-1539" src="/assets/img/posts/2019/01/Using-the-ITestOutputHelper.jpg" alt="Using the ITestOutputHelper" width="371" height="125" /></a>
  
  <p id="caption-attachment-1539" class="wp-caption-text">
    Using the ITestOutputHelper
  </p>
</div>

Next, use the WriteLine method of the ITestOutputHelper object to create the desired output.

<div id="attachment_1540" style="width: 519px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Creating-a-custom-message-for-the-test-result.jpg"><img aria-describedby="caption-attachment-1540" loading="lazy" class="size-full wp-image-1540" src="/assets/img/posts/2019/01/Creating-a-custom-message-for-the-test-result.jpg" alt="Creating a custom message for the test result" width="509" height="170" /></a>
  
  <p id="caption-attachment-1540" class="wp-caption-text">
    Creating a custom message for the test result
  </p>
</div>

When you run the test, you will see the message in the test result window.

<div id="attachment_1541" style="width: 702px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Output-message-of-the-test.jpg"><img aria-describedby="caption-attachment-1541" loading="lazy" class="size-full wp-image-1541" src="/assets/img/posts/2019/01/Output-message-of-the-test.jpg" alt="Output message of the test" width="692" height="110" /></a>
  
  <p id="caption-attachment-1541" class="wp-caption-text">
    Output message of the test
  </p>
</div>

## Share resources over multiple tests

Previously, I mentioned that for every test a new object is instantiated and therefore isolated from the other tests.  Sometimes you need to share a resource with several tests. This can be done with Fixtures. First, you have to create a so-called fixture class with the information you want to share. In my simple example, I set DateTime.Now to demonstrate that every test uses the same instance of the object.

<div id="attachment_1542" style="width: 312px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Fixture-class-to-share-the-time-property.jpg"><img aria-describedby="caption-attachment-1542" loading="lazy" class="size-full wp-image-1542" src="/assets/img/posts/2019/01/Fixture-class-to-share-the-time-property.jpg" alt="Fixture class to share the time property" width="302" height="173" /></a>
  
  <p id="caption-attachment-1542" class="wp-caption-text">
    Fixture class to share the time property
  </p>
</div>

Next, I am creating a collection class with the CollectionDefiniton attribute and the ICollectionFixture interface with my previously created fixture class.

<div id="attachment_1543" style="width: 453px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Creating-the-collection-to-share-date-accross-tests.jpg"><img aria-describedby="caption-attachment-1543" loading="lazy" class="wp-image-1543 size-full" src="/assets/img/posts/2019/01/Creating-the-collection-to-share-date-accross-tests.jpg" alt="Creating the collection to share date across tests" width="443" height="90" /></a>
  
  <p id="caption-attachment-1543" class="wp-caption-text">
    Creating the collection to share date across tests
  </p>
</div>

Finally, I add the Collection attribute with the previously set name to my test class and pass the fixture class in the constructor.

<div id="attachment_1544" style="width: 588px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Passing-the-collection-to-the-test-class.jpg"><img aria-describedby="caption-attachment-1544" loading="lazy" class="size-full wp-image-1544" src="/assets/img/posts/2019/01/Passing-the-collection-to-the-test-class.jpg" alt="Passing the collection to the test class" width="578" height="263" /></a>
  
  <p id="caption-attachment-1544" class="wp-caption-text">
    Passing the collection to the test class
  </p>
</div>

To demonstrate that the _timeFixture object stays the same, I run a couple of tests with Thread.Sleep(1500) and both tests will output the same time.

<div id="attachment_1546" style="width: 423px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Tests-using-the-fixture-object.jpg"><img aria-describedby="caption-attachment-1546" loading="lazy" class="size-full wp-image-1546" src="/assets/img/posts/2019/01/Tests-using-the-fixture-object.jpg" alt="Tests using the fixture object" width="413" height="274" /></a>
  
  <p id="caption-attachment-1546" class="wp-caption-text">
    Tests using the fixture object
  </p>
</div>

Both tests will print the same output.

<div id="attachment_1545" style="width: 290px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/The-same-output-of-the-fixture-object.jpg"><img aria-describedby="caption-attachment-1545" loading="lazy" class="size-full wp-image-1545" src="/assets/img/posts/2019/01/The-same-output-of-the-fixture-object.jpg" alt="The same output of the fixture object" width="280" height="166" /></a>
  
  <p id="caption-attachment-1545" class="wp-caption-text">
    The same output of the fixture object
  </p>
</div>

## Provide test data from a class

Previously, I showed how to use the Theory attribute to pass several parameters for the test. If you want the same data for several tests, you would have to enter it several times. This is error-prone and unpractical. Therefore, you can place these values in a class and just add a reference to the class.

Create a new class with a static property and only a getter which yield returns all your test data as object arrays.

<div id="attachment_1547" style="width: 409px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Class-with-test-data.jpg"><img aria-describedby="caption-attachment-1547" loading="lazy" class="size-full wp-image-1547" src="/assets/img/posts/2019/01/Class-with-test-data.jpg" alt="Class with test data" width="399" height="257" /></a>
  
  <p id="caption-attachment-1547" class="wp-caption-text">
    Class with test data
  </p>
</div>

For your test, use the MemberData instead of the InlineData attribute and provide the name of the property and the type of the class containing your test data.

<div id="attachment_1548" style="width: 664px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Using-test-data-from-a-class.jpg"><img aria-describedby="caption-attachment-1548" loading="lazy" class="size-full wp-image-1548" src="/assets/img/posts/2019/01/Using-test-data-from-a-class.jpg" alt="Using test data from a class" width="654" height="154" /></a>
  
  <p id="caption-attachment-1548" class="wp-caption-text">
    Using test data from a class
  </p>
</div>

## Provide test data with a custom attribute

A custom attribute works the same way as the MemberData attribute but it is even less to write in your test. Create a new class and inherit from the DataAttribute class. Then override the GetData method.

<div id="attachment_1549" style="width: 526px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Override-the-GetData-method-from-the-DataAttribute-class.jpg"><img aria-describedby="caption-attachment-1549" loading="lazy" class="size-full wp-image-1549" src="/assets/img/posts/2019/01/Override-the-GetData-method-from-the-DataAttribute-class.jpg" alt="Override the GetData method from the DataAttribute class" width="516" height="194" /></a>
  
  <p id="caption-attachment-1549" class="wp-caption-text">
    Override the GetData method from the DataAttribute class
  </p>
</div>

After you created the class, add the name of the class (without Attribute) as the attribute to your Theory. xUnit will recognize your attribute and call the GetData method.

<div id="attachment_1550" style="width: 554px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Use-the-custom-attribute.jpg"><img aria-describedby="caption-attachment-1550" loading="lazy" class="size-full wp-image-1550" src="/assets/img/posts/2019/01/Use-the-custom-attribute.jpg" alt="Use the custom attribute" width="544" height="152" /></a>
  
  <p id="caption-attachment-1550" class="wp-caption-text">
    Use the custom attribute
  </p>
</div>

## Provide test data from an external source

The last method to provide data for your tests is from an external source. To read the data from a csv file, I placed the csv file in the root folder of my project and created a class with a static property. In the getter of the property, I read the file, split the values and return them as object arrays. Don&#8217;t forget to set the Copy to Output Directory property of the csv file to Copy always or Copy if newer. Otherwise, the file won&#8217;t be copied when you compile your code and therefore won&#8217;t be found at runtime.

<div id="attachment_1551" style="width: 659px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Read-data-from-a-csv-file.jpg"><img aria-describedby="caption-attachment-1551" loading="lazy" class="size-full wp-image-1551" src="/assets/img/posts/2019/01/Read-data-from-a-csv-file.jpg" alt="Read data from a csv file" width="649" height="216" /></a>
  
  <p id="caption-attachment-1551" class="wp-caption-text">
    Read data from a csv file
  </p>
</div>

Now use the MemberData attribute for your test to add the name of the property and the type of your class.

<div id="attachment_1552" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/01/Provide-data-from-an-external-file-to-your-test.jpg"><img aria-describedby="caption-attachment-1552" loading="lazy" class="wp-image-1552" src="/assets/img/posts/2019/01/Provide-data-from-an-external-file-to-your-test.jpg" alt="Provide data from an external file to your test" width="700" height="134" /></a>
  
  <p id="caption-attachment-1552" class="wp-caption-text">
    Provide data from an external file to your test
  </p>
</div>

## Conclusion

In this post, I gave a quick overview of xUnit and explained how to get data from several sources and how to reduce duplicate code. For more information on xUnit, I can recommend the Pluralsight course &#8220;<a href="https://app.pluralsight.com/library/courses/dotnet-core-testing-code-xunit-dotnet-getting-started/table-of-contents" target="_blank" rel="noopener">Testing .NET Core Code with xUnit.net: Getting Started</a>&#8221; from Jason Robert.

You can find the code of my demo on <a href="https://github.com/WolfgangOfner/xUnit-Getting-Started" target="_blank" rel="noreferrer noopener" aria-label="You can find all my solutions on GitHub. (opens in a new tab)">GitHub</a>.