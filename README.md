# LinearProgramming

## Goal

The Swarthmore Computer Science department has extremely high demand for
courses. As a result, not everyone who wants to take certain courses in the
department is able to.

The current system to make these decisions is essentially manual, and is extremely
difficult and time consuming. This application provides a (relatively) easy to
use interface to allow people without experience in Linear Programming to decide
who gets accepted into the course.

You can define an arbitrary number of constraints that can be either numeric
(such as "I can only have N students in the class", or "I want at least M first year
students to take the course"), or percentage (such as "I want men to make up at most
60 percent of the course").

The parser will automatically take this plaintext file and formulate it into
a linear program, which is then solved using scipy's linear program solver.

## Future Goals

This is very much a work in progress, but I hope to extend this to allow a
richer set of goals. Currently, goals are prioritized one after another (meaning
you must fully optimize the first goal before you can optimize the next). It
might be interesting to allow more variability with this. In addition, I'd like
to have a seperate .csv parser to automatically populate the attributes array.

Hopefully, the department will find this code to be of use :)
