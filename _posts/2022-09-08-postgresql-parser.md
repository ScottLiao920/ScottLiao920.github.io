---
title: PostgreSQL Parser
---

# General Idea
* PostgreSQL parsing stage does more than normal tokenizing and parsing, it also creates Query structures for the various complex queries that are passed to the optimizer and then executor.
* Utilize Bison and Flex for parsing and lexing input string
 
# Parsing
* A lexer (defined in *scan.l*) recognizes identifiers, SQL keywords and so on. For each identifier or keyword find, a *token *is generated  and fed to the parser.
* Paser (defined in *gram.y*) consists of a set of grammar rules and actions to build up a parse tree.

# Transformation
* Interpre the parse tree to understand tables, functions and operators referenced, this semantically intepreted structure is called a query tree.
* Seperate raw parsing and semantical analysis so that control  commands (`BEGIN`, `ROLLBACK`, ...) can be executed without further analysis.
