# Dictate ASR model benchmarks

Generated 2026-10-22T21:33:54+00:00 on AsusGL552 (Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 8 logical CPUs, 15.5 GiB RAM).

Tree: `d5ad53c1cd65` (main, dirty), dirty-tree SHA-256 `22498226010831d5`.
Binary: `/home/saeid/projects/dictate/target/release/dictate` (SHA-256 `1c75c466d4a34ae5`). Corpus: `/home/saeid/.cache/dictate-benchmarks/2026-10-22/benchmark-corpus.json` (SHA-256 `a654316a897a4cd3`).

> One engine per process, loaded once and kept resident. JFK runs are warmed and unpaced; streaming paced runs feed audio at real time. RTF is processing time divided by audio time. RSS is the whole process from `/proc/self/status`, not weights-only memory. The 12-clip LibriSpeech subset is small read English, not conversational or accent coverage.

## Summary

| Engine | Threads | Status | Load ms | First JFK decode ms | JFK median ms | JFK RTF | Corpus WER | CER | Peak RSS | Paced stop ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard | default | ok | 2864 | 719 | 671 | 0.061 | 1.50% | 0.11% | 1.7 GiB | - |
| standard | 1 | ok | 2768 | 1775 | 1764 | 0.160 | 1.50% | 0.11% | 1.7 GiB | - |
| standard | 2 | ok | 2867 | 1118 | 1106 | 0.101 | 1.50% | 0.11% | 1.7 GiB | - |
| fluid | default | ok | 390 | 1308 | 1309 | 0.119 | 2.00% | 0.21% | 521.4 MiB | - |
| fluid | 1 | ok | 385 | 3915 | 3879 | 0.353 | 2.00% | 0.21% | 520.2 MiB | - |
| fluid | 2 | ok | 384 | 2274 | 2248 | 0.204 | 2.00% | 0.21% | 521.9 MiB | - |
| whisper | default | ok | 40 | 1687 | 1626 | 0.148 | 5.00% | 1.60% | 246.2 MiB | - |
| whisper | 1 | ok | 40 | 2964 | 2947 | 0.268 | 5.00% | 1.60% | 246.1 MiB | - |
| whisper | 2 | ok | 42 | 1656 | 1639 | 0.149 | 5.00% | 1.60% | 245.9 MiB | - |
| live | default | ok | 3174 | 2514 | 2490 | 0.226 | 1.50% | 0.11% | 1.5 GiB | 526 |
| live | 1 | ok | 2928 | 5071 | 5244 | 0.477 | 1.50% | 0.11% | 1.5 GiB | 917 |
| live | 2 | ok | 2971 | 3203 | 3233 | 0.294 | 1.50% | 0.11% | 1.5 GiB | 582 |
| moonshine | 1 | ok | 97 | 1668 | 1583 | 0.144 | 6.00% | 2.77% | 218.1 MiB | 127 |
| moonshine | runtime-default | ok | 111 | 1912 | 1955 | 0.178 | 6.00% | 2.77% | 218.7 MiB | 149 |

## Quality per clip

| Engine | Threads | Clip | Duration s | Decode ms | WER | S | D | I |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard | default | 6930-75918-0000 | 3.505 | 252 | 0.0% | 0 | 0 | 0 |
| standard | default | 6930-75918-0002 | 5.025 | 344 | 0.0% | 0 | 0 | 0 |
| standard | default | 260-123288-0018 | 3.245 | 238 | 0.0% | 0 | 0 | 0 |
| standard | default | 260-123288-0022 | 3.705 | 287 | 0.0% | 0 | 0 | 0 |
| standard | default | 2961-960-0004 | 8.220 | 543 | 10.5% | 1 | 1 | 0 |
| standard | default | 2961-960-0005 | 10.355 | 637 | 0.0% | 0 | 0 | 0 |
| standard | default | 1995-1826-0000 | 9.485 | 577 | 3.7% | 1 | 0 | 0 |
| standard | default | 1995-1826-0001 | 10.170 | 619 | 0.0% | 0 | 0 | 0 |
| standard | default | 8455-210777-0028 | 7.735 | 521 | 0.0% | 0 | 0 | 0 |
| standard | default | 8455-210777-0029 | 3.085 | 224 | 0.0% | 0 | 0 | 0 |
| standard | default | 4507-16021-0033 | 5.545 | 378 | 0.0% | 0 | 0 | 0 |
| standard | default | 4507-16021-0034 | 11.270 | 673 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 6930-75918-0000 | 3.505 | 603 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 6930-75918-0002 | 5.025 | 839 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 260-123288-0018 | 3.245 | 568 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 260-123288-0022 | 3.705 | 646 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 2961-960-0004 | 8.220 | 1389 | 10.5% | 1 | 1 | 0 |
| standard | 1 | 2961-960-0005 | 10.355 | 1717 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 1995-1826-0000 | 9.485 | 1532 | 3.7% | 1 | 0 | 0 |
| standard | 1 | 1995-1826-0001 | 10.170 | 1651 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 8455-210777-0028 | 7.735 | 1255 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 8455-210777-0029 | 3.085 | 524 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 4507-16021-0033 | 5.545 | 919 | 0.0% | 0 | 0 | 0 |
| standard | 1 | 4507-16021-0034 | 11.270 | 2088 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 6930-75918-0000 | 3.505 | 368 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 6930-75918-0002 | 5.025 | 523 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 260-123288-0018 | 3.245 | 350 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 260-123288-0022 | 3.705 | 386 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 2961-960-0004 | 8.220 | 806 | 10.5% | 1 | 1 | 0 |
| standard | 2 | 2961-960-0005 | 10.355 | 1010 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 1995-1826-0000 | 9.485 | 934 | 3.7% | 1 | 0 | 0 |
| standard | 2 | 1995-1826-0001 | 10.170 | 989 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 8455-210777-0028 | 7.735 | 768 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 8455-210777-0029 | 3.085 | 326 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 4507-16021-0033 | 5.545 | 569 | 0.0% | 0 | 0 | 0 |
| standard | 2 | 4507-16021-0034 | 11.270 | 1163 | 0.0% | 0 | 0 | 0 |
| fluid | default | 6930-75918-0000 | 3.505 | 745 | 12.5% | 1 | 0 | 0 |
| fluid | default | 6930-75918-0002 | 5.025 | 879 | 0.0% | 0 | 0 | 0 |
| fluid | default | 260-123288-0018 | 3.245 | 743 | 0.0% | 0 | 0 | 0 |
| fluid | default | 260-123288-0022 | 3.705 | 833 | 0.0% | 0 | 0 | 0 |
| fluid | default | 2961-960-0004 | 8.220 | 1157 | 10.5% | 1 | 1 | 0 |
| fluid | default | 2961-960-0005 | 10.355 | 1450 | 0.0% | 0 | 0 | 0 |
| fluid | default | 1995-1826-0000 | 9.485 | 1351 | 3.7% | 1 | 0 | 0 |
| fluid | default | 1995-1826-0001 | 10.170 | 1924 | 0.0% | 0 | 0 | 0 |
| fluid | default | 8455-210777-0028 | 7.735 | 1190 | 0.0% | 0 | 0 | 0 |
| fluid | default | 8455-210777-0029 | 3.085 | 727 | 0.0% | 0 | 0 | 0 |
| fluid | default | 4507-16021-0033 | 5.545 | 911 | 0.0% | 0 | 0 | 0 |
| fluid | default | 4507-16021-0034 | 11.270 | 1379 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 6930-75918-0000 | 3.505 | 2498 | 12.5% | 1 | 0 | 0 |
| fluid | 1 | 6930-75918-0002 | 5.025 | 2868 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 260-123288-0018 | 3.245 | 2376 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 260-123288-0022 | 3.705 | 2491 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 2961-960-0004 | 8.220 | 3431 | 10.5% | 1 | 1 | 0 |
| fluid | 1 | 2961-960-0005 | 10.355 | 3719 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 1995-1826-0000 | 9.485 | 3611 | 3.7% | 1 | 0 | 0 |
| fluid | 1 | 1995-1826-0001 | 10.170 | 3742 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 8455-210777-0028 | 7.735 | 3244 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 8455-210777-0029 | 3.085 | 2339 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 4507-16021-0033 | 5.545 | 2792 | 0.0% | 0 | 0 | 0 |
| fluid | 1 | 4507-16021-0034 | 11.270 | 3935 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 6930-75918-0000 | 3.505 | 1367 | 12.5% | 1 | 0 | 0 |
| fluid | 2 | 6930-75918-0002 | 5.025 | 1605 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 260-123288-0018 | 3.245 | 1369 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 260-123288-0022 | 3.705 | 1431 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 2961-960-0004 | 8.220 | 1945 | 10.5% | 1 | 1 | 0 |
| fluid | 2 | 2961-960-0005 | 10.355 | 2206 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 1995-1826-0000 | 9.485 | 2117 | 3.7% | 1 | 0 | 0 |
| fluid | 2 | 1995-1826-0001 | 10.170 | 2207 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 8455-210777-0028 | 7.735 | 1899 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 8455-210777-0029 | 3.085 | 1330 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 4507-16021-0033 | 5.545 | 1624 | 0.0% | 0 | 0 | 0 |
| fluid | 2 | 4507-16021-0034 | 11.270 | 2284 | 0.0% | 0 | 0 | 0 |
| whisper | default | 6930-75918-0000 | 3.505 | 1181 | 0.0% | 0 | 0 | 0 |
| whisper | default | 6930-75918-0002 | 5.025 | 1231 | 0.0% | 0 | 0 | 0 |
| whisper | default | 260-123288-0018 | 3.245 | 1207 | 22.2% | 1 | 1 | 0 |
| whisper | default | 260-123288-0022 | 3.705 | 1213 | 0.0% | 0 | 0 | 0 |
| whisper | default | 2961-960-0004 | 8.220 | 1305 | 15.8% | 2 | 1 | 0 |
| whisper | default | 2961-960-0005 | 10.355 | 1516 | 0.0% | 0 | 0 | 0 |
| whisper | default | 1995-1826-0000 | 9.485 | 1401 | 11.1% | 3 | 0 | 0 |
| whisper | default | 1995-1826-0001 | 10.170 | 1537 | 0.0% | 0 | 0 | 0 |
| whisper | default | 8455-210777-0028 | 7.735 | 1225 | 11.8% | 1 | 0 | 1 |
| whisper | default | 8455-210777-0029 | 3.085 | 1175 | 0.0% | 0 | 0 | 0 |
| whisper | default | 4507-16021-0033 | 5.545 | 1211 | 0.0% | 0 | 0 | 0 |
| whisper | default | 4507-16021-0034 | 11.270 | 1706 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 6930-75918-0000 | 3.505 | 2136 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 6930-75918-0002 | 5.025 | 2190 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 260-123288-0018 | 3.245 | 2195 | 22.2% | 1 | 1 | 0 |
| whisper | 1 | 260-123288-0022 | 3.705 | 2183 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 2961-960-0004 | 8.220 | 2299 | 15.8% | 2 | 1 | 0 |
| whisper | 1 | 2961-960-0005 | 10.355 | 2746 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 1995-1826-0000 | 9.485 | 2499 | 11.1% | 3 | 0 | 0 |
| whisper | 1 | 1995-1826-0001 | 10.170 | 2718 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 8455-210777-0028 | 7.735 | 2245 | 11.8% | 1 | 0 | 1 |
| whisper | 1 | 8455-210777-0029 | 3.085 | 2134 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 4507-16021-0033 | 5.545 | 2191 | 0.0% | 0 | 0 | 0 |
| whisper | 1 | 4507-16021-0034 | 11.270 | 3134 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 6930-75918-0000 | 3.505 | 1170 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 6930-75918-0002 | 5.025 | 1214 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 260-123288-0018 | 3.245 | 1162 | 22.2% | 1 | 1 | 0 |
| whisper | 2 | 260-123288-0022 | 3.705 | 1231 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 2961-960-0004 | 8.220 | 1271 | 15.8% | 2 | 1 | 0 |
| whisper | 2 | 2961-960-0005 | 10.355 | 1533 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 1995-1826-0000 | 9.485 | 1401 | 11.1% | 3 | 0 | 0 |
| whisper | 2 | 1995-1826-0001 | 10.170 | 1522 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 8455-210777-0028 | 7.735 | 1241 | 11.8% | 1 | 0 | 1 |
| whisper | 2 | 8455-210777-0029 | 3.085 | 1181 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 4507-16021-0033 | 5.545 | 1222 | 0.0% | 0 | 0 | 0 |
| whisper | 2 | 4507-16021-0034 | 11.270 | 1716 | 0.0% | 0 | 0 | 0 |
| live | default | 6930-75918-0000 | 3.505 | 1069 | 0.0% | 0 | 0 | 0 |
| live | default | 6930-75918-0002 | 5.025 | 1274 | 0.0% | 0 | 0 | 0 |
| live | default | 260-123288-0018 | 3.245 | 974 | 0.0% | 0 | 0 | 0 |
| live | default | 260-123288-0022 | 3.705 | 1068 | 0.0% | 0 | 0 | 0 |
| live | default | 2961-960-0004 | 8.220 | 1943 | 15.8% | 2 | 1 | 0 |
| live | default | 2961-960-0005 | 10.355 | 2357 | 0.0% | 0 | 0 | 0 |
| live | default | 1995-1826-0000 | 9.485 | 2156 | 0.0% | 0 | 0 | 0 |
| live | default | 1995-1826-0001 | 10.170 | 2400 | 0.0% | 0 | 0 | 0 |
| live | default | 8455-210777-0028 | 7.735 | 1845 | 0.0% | 0 | 0 | 0 |
| live | default | 8455-210777-0029 | 3.085 | 959 | 0.0% | 0 | 0 | 0 |
| live | default | 4507-16021-0033 | 5.545 | 1430 | 0.0% | 0 | 0 | 0 |
| live | default | 4507-16021-0034 | 11.270 | 2586 | 0.0% | 0 | 0 | 0 |
| live | 1 | 6930-75918-0000 | 3.505 | 2286 | 0.0% | 0 | 0 | 0 |
| live | 1 | 6930-75918-0002 | 5.025 | 2774 | 0.0% | 0 | 0 | 0 |
| live | 1 | 260-123288-0018 | 3.245 | 2054 | 0.0% | 0 | 0 | 0 |
| live | 1 | 260-123288-0022 | 3.705 | 2300 | 0.0% | 0 | 0 | 0 |
| live | 1 | 2961-960-0004 | 8.220 | 4153 | 15.8% | 2 | 1 | 0 |
| live | 1 | 2961-960-0005 | 10.355 | 5106 | 0.0% | 0 | 0 | 0 |
| live | 1 | 1995-1826-0000 | 9.485 | 4573 | 0.0% | 0 | 0 | 0 |
| live | 1 | 1995-1826-0001 | 10.170 | 5050 | 0.0% | 0 | 0 | 0 |
| live | 1 | 8455-210777-0028 | 7.735 | 3927 | 0.0% | 0 | 0 | 0 |
| live | 1 | 8455-210777-0029 | 3.085 | 2043 | 0.0% | 0 | 0 | 0 |
| live | 1 | 4507-16021-0033 | 5.545 | 2853 | 0.0% | 0 | 0 | 0 |
| live | 1 | 4507-16021-0034 | 11.270 | 5335 | 0.0% | 0 | 0 | 0 |
| live | 2 | 6930-75918-0000 | 3.505 | 1486 | 0.0% | 0 | 0 | 0 |
| live | 2 | 6930-75918-0002 | 5.025 | 1732 | 0.0% | 0 | 0 | 0 |
| live | 2 | 260-123288-0018 | 3.245 | 1258 | 0.0% | 0 | 0 | 0 |
| live | 2 | 260-123288-0022 | 3.705 | 1391 | 0.0% | 0 | 0 | 0 |
| live | 2 | 2961-960-0004 | 8.220 | 2540 | 15.8% | 2 | 1 | 0 |
| live | 2 | 2961-960-0005 | 10.355 | 3095 | 0.0% | 0 | 0 | 0 |
| live | 2 | 1995-1826-0000 | 9.485 | 2811 | 0.0% | 0 | 0 | 0 |
| live | 2 | 1995-1826-0001 | 10.170 | 3107 | 0.0% | 0 | 0 | 0 |
| live | 2 | 8455-210777-0028 | 7.735 | 2429 | 0.0% | 0 | 0 | 0 |
| live | 2 | 8455-210777-0029 | 3.085 | 1315 | 0.0% | 0 | 0 | 0 |
| live | 2 | 4507-16021-0033 | 5.545 | 1860 | 0.0% | 0 | 0 | 0 |
| live | 2 | 4507-16021-0034 | 11.270 | 3371 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 6930-75918-0000 | 3.505 | 637 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 6930-75918-0002 | 5.025 | 1005 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 260-123288-0018 | 3.245 | 605 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 260-123288-0022 | 3.705 | 639 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 2961-960-0004 | 8.220 | 1912 | 42.1% | 6 | 2 | 0 |
| moonshine | 1 | 2961-960-0005 | 10.355 | 1662 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 1995-1826-0000 | 9.485 | 2082 | 7.4% | 2 | 0 | 0 |
| moonshine | 1 | 1995-1826-0001 | 10.170 | 1913 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 8455-210777-0028 | 7.735 | 1190 | 11.8% | 1 | 0 | 1 |
| moonshine | 1 | 8455-210777-0029 | 3.085 | 425 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 4507-16021-0033 | 5.545 | 1079 | 0.0% | 0 | 0 | 0 |
| moonshine | 1 | 4507-16021-0034 | 11.270 | 2011 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 6930-75918-0000 | 3.505 | 805 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 6930-75918-0002 | 5.025 | 1263 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 260-123288-0018 | 3.245 | 674 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 260-123288-0022 | 3.705 | 790 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 2961-960-0004 | 8.220 | 2367 | 42.1% | 6 | 2 | 0 |
| moonshine | runtime-default | 2961-960-0005 | 10.355 | 2115 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 1995-1826-0000 | 9.485 | 2497 | 7.4% | 2 | 0 | 0 |
| moonshine | runtime-default | 1995-1826-0001 | 10.170 | 2373 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 8455-210777-0028 | 7.735 | 1521 | 11.8% | 1 | 0 | 1 |
| moonshine | runtime-default | 8455-210777-0029 | 3.085 | 566 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 4507-16021-0033 | 5.545 | 1298 | 0.0% | 0 | 0 | 0 |
| moonshine | runtime-default | 4507-16021-0034 | 11.270 | 2508 | 0.0% | 0 | 0 | 0 |

## Stress and false words

| Engine | Threads | 5 s silence words | 5 s noise words | Paced stop ms | Max backlog ms |
| --- | --- | --- | --- | --- | --- |
| standard | default | 0 | 0 | - | - |
| standard | 1 | 0 | 0 | - | - |
| standard | 2 | 0 | 0 | - | - |
| fluid | default | 0 | 0 | - | - |
| fluid | 1 | 0 | 0 | - | - |
| fluid | 2 | 0 | 0 | - | - |
| whisper | default | 0 | 0 | - | - |
| whisper | 1 | 0 | 0 | - | - |
| whisper | 2 | 0 | 0 | - | - |
| live | default | 0 | 0 | 526 | 0 |
| live | 1 | 0 | 0 | 917 | 0 |
| live | 2 | 0 | 0 | 582 | 0 |
| moonshine | 1 | 0 | 0 | 127 | 83 |
| moonshine | runtime-default | 0 | 0 | 149 | 154 |

## Method

- One process per engine/thread configuration, model loaded once. The standard, Fluid, Live and Whisper paths use their native 1/2-thread constructor settings; the default setting is the app's real default. Moonshine exposes only single-thread mode or the ONNX Runtime default, so it is never reported as an exact two-thread run.
- Audio is preloaded before timers start. One warmup JFK decode is excluded from the measured JFK runs; it covers the whole first recording for every engine, and streaming also reports the first chunk separately. Five warmed JFK runs by default; the 12-clip corpus runs once after warmup.
- WER normalization: NFKC, case-fold, remove bracketed non-speech tags, hyphens to spaces, keep alphanumerics and spaces, delete other punctuation (apostrophes included). Corpus WER is total edits / total reference words; S/D/I are from a word Levenshtein alignment. CER uses the same normalization without spaces. Numbers are not expanded.
- Silence and noise false words are reported separately and are not part of corpus WER. Synthetic JFK repeats are labeled synthetic and never enter corpus WER.
- Live and Moonshine paced runs feed 100 ms (Moonshine) or the model's native chunk (Live) at real-time arrival, then flush exactly as the production lifecycle does. Slower-than-real-time is recorded, not gated. `stop_to_final_ms` includes the flush/finalization compute.
- Measurements are one machine, warm file cache, no GUI latency, and no cold-disk claim. CPU time is parent-side `RUSAGE_CHILDREN`; peak RSS comes from the child. Thermal state, load, and CPU governor can move results between runs.

## Files

- `benchmark.json`: full raw documents and summaries.
- `benchmark.csv`: one summary row per configuration.
- `benchmark-clips.csv`: per-clip references, hypotheses, WER, CER, S/D/I.
- `benchmark-stress.csv`: silence/noise/JFK stress transcripts.
- `benchmark.html`: self-contained, offline report with engine/thread filters.
- `raw/`: one stdout JSON and stderr log per process.
