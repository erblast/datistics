


# Level of detail expressions

It happens often that you want to create calculated fields that aggregate data but you need them to be independent of the dimensions in your figure. You can do this by using the level of detail expressions.


## Fixed
Use a fixed set of dimensions

Total population per country:
``` 
{FIXED continent, country : SUM(population) }
```

## Exclude
Ignore a specific dimension in the vis

Total population per country:
``` 
{EXCLUDE city : SUM(population)}
```

## Include
Add a specific dimension in the vis

Total population per country:
```
{INCLUDE country : SUM(population)}
```

# Min(1) trick
http://datapsientist.blogspot.com/2014/10/tableau-trick-for-work-min1.html

# Creating a Parameter Control to Sort By Dimension 
https://kb.tableau.com/articles/howto/creating-a-parameter-control-to-sort-by-dimension

# Send parameters from workbook to workbook via URL

https://community.tableau.com/thread/202493

# Workaround for Nested Containers

https://community.tableau.com/thread/178426
