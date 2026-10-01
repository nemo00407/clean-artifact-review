# E10 - Mutation testing of the confinement audit

Each mutant is a runtime module carrying exactly one policy violation. The unmutated runtime reports **0** violations.

| violation class | audit verdict |
|---|---|
| forbidden import | caught |
| forbidden from-import | caught |
| ambient builtin: eval | caught |
| ambient builtin: getattr | caught |
| file open outside the gateway | caught |
| array-library filesystem API | caught |
| array-library FS via from-import | caught |
| untrusted backend outside the read path | caught |
| storage path outside the boundary | caught |
| attribute-chain escape | missed |
| alias rebinding of the array library | missed |

Caught 9 of 11 classes. The 2 misses are recorded rather than removed from the suite: the audit is a conservative syntactic check over the runtime's own modules and does not reason about dynamic rebinding or attribute chains that name no forbidden identifier. What it establishes is that a change opening a second channel of the caught classes stops at the build.
