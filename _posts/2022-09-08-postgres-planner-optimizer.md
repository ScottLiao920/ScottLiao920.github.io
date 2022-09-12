---
title: Available Join/Scan Methods in PostgreSQL
---

## Basics: PostgreSQL filesystem

* PostgreSQL is a disk-oriented database, which means its data are stored on disks in pages. Access to heap page can
  be random or sequential, where random access costs roughly ~4times more expensive than a sequential access.
* PostgreSQL is row-based, which means all columns of a tuple is stored on the same page. To address the issue of
  variable-length data, tuples data are stored from back to beginning while their pointers are stored from beginning to
  back in a heap page. ![Heap Page in PostgreSQL]({{ BASE_PATH }}/pictures/HeapPage.png)
* Indexes are stored on separate pages. An entry of index table consists of (key value, TID of corresponding tuple).
* Tuple Identifier (TID) is used to record the unique location of a tuple. It is 6-byte long and consists of two parts:
  4-byte-long page number and 4-byte-long index number inside the page.

## Scan Methods

* **Sequential Scan**
  * Iterate through the table sequentially and return tuples that matches predicate one at a time.
  * Cost is computed based on seq_page_cost * no. of pages.
    * More precisely,

        `cost = blocks * seq_page_cost +
         number_of_records * cpu_tuple_cost +
         number_of_records * cpu_filter_cost`
  * Ideal for high-selectivity operations.
* **Index Scan**
  * Get the TID of tuples that match the predicate, access corresponding heap pages to get the tuple.
  * Cost is computed based on 2 * random_page_cost (index + heap).
  * Ideal for low-selectivity operations.
* **Index Only Scan**
  * Similar to index scan but no need to access heap pages because only the column with index is required to emit.
  * Example: `select num from table where num=1;`
* **Bitmap Scan**
  * Mix-up of index scan and sequential scan. Two-phases: bitmap index scan and bitmap heap scan.
  * Bitmap index scan: Read all indexes to create a bitmap of tuple TIDs. Each entry in bitmap is page no. and it
      consists of indexes of tuples to read in that page.
  * Bitmap heap scan: Read tuples based on the bitmap.
  * Ideal for the most cases where sensitivity is neither too high nor too low
* **TID Scan**
  * Only applicable when TID is the only predicate.
  * Example `select * from table where ctid=(page, tuple);`

## Join Methods

* **Nested Loop Join**
  * pseudocode for join table A & table B on predicate `A.ID < B.ID`:

        for every tuple a in A:
            for evey tuple b in B:
                if (a.ID < b.ID):
                    emit tuple (a, b)
  
* **Merge Join**
  * Only applicable to equi-join with predicate on indexed columns

         For each tuple r in A
             For each tuple s in B
                 If (r.ID = s.ID)
                     Emit output tuple (r,s)
                     Break;
             If (r.ID > s.ID)
                  # move to next s
                  Continue;
             Else
                  Break;
* **Hash Join**
  * Only applicable to equi-join
  * Two Phases:
    * Build-phase: build hash table on inner table
    * Probe-phase: probe the outer table, emit joined tuples that match predicate
  * Pseudocode (A join B on `A.ID = B.ID`:

        # Build Phase
        for every tuple r in B:
            insert h(r.ID) into hash table 
        # Probe Phase
        for every tuple s in A:
            for every tuple r in bucket h(s.ID):
                emit (r, s)

### Logical Joins

* **Inner Join**
  * Matched tuple
* **Outer Join**
  * Left-outer/Right-outer select concatenated tuples from all of left(right) tables and matched tuples from right(left) tables.
  * Output dimension is the same as left(right) table.
* **Semi Join**
  * Not really a join as it only uses left/right table for refernce, only emits part of left/right tuple.
* **Anti Join**
  * Not-matched tuple
* As in the figure below ![Joins]({{ BASE_PATH }}/pictures/Joins.png)