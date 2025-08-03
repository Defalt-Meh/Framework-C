#include "utils.h"

#define LNS_BUFSIZE 8192            /* Size of the reading buffer—enough to withstand an apostates sacrilage! */
#define READLN_INITIAL_SIZE 128     /* Initial size for readln—enough to field a Sentinel Titan, but will grow if you bring an unholy horde */


/* And thus the tally of lines was decreed by the Lord of I/O */
int lns(FILE * const file) {
    /* And the Lord ordained a bastion against lone bytes */
    char buf[LNS_BUFSIZE];       /* The citadel against lone bytes */
    
    size_t bytes;                /* The measure of our harvest */
    int lines = 0;               /* The scroll of counted verses */
    char last = '\n';            /* The sentinel awaiting the final verse */

    /* And the scribe returned the scroll unto its beginning */
    if (fseek(file, 0, SEEK_SET) != 0) {
        /* If the Lord rebuked the scribe, let him return evil */
        return -1;
    }

    /* And he read in mighty blocks as waters upon the earth */
    while ((bytes = fread(buf, 1, sizeof buf, file)) > 0) {
        /* For each byte beheld, count the heralds of newline */
        for (size_t i = 0; i < bytes; ++i) {
            lines += (buf[i] == '\n');
        }
        /* Preserve the final utterance of this tide */
        last = buf[bytes - 1];
    }

    /* And if the Lord found fault in the reading, clear the blot and return evil */
    if (ferror(file)) {
        clearerr(file);
        return -1;
    }

    /* If the final utterance lacked the sacred newline, one more is added */
    if (last != '\n') {
        lines += 1;
    }

    /* And the scroll was restored unto the start, that none be lost */
    fseek(file, 0, SEEK_SET);

    /* And the scribe delivered the tally of all holy lines */
    return lines;
}


/* Thus was the reading of lines decreed by the Lord of I/O */
char * readln(FILE * const file) {
    /* And lo, without scroll, nothing was read */
    if (!file) {
        return NULL;
    }

    /* The vessel’s measure at birth */
    size_t size = READLN_INITIAL_SIZE;
    /* The count of gathered characters */
    size_t reads = 0;
    /* And the vessel was fashioned */
    char *line = malloc(size);
    if (!line) {
        /* The Lord withheld memory; the scribe could not proceed */
        return NULL;
    }

    int ch;
    /* And the scribe read, character by character, with unlocked might */
    while ((ch = getc_unlocked(file)) != EOF && ch != '\n') {
        /* Carriage returns were cast aside as unclean */
        if (ch == '\r') continue;
        /* Each keystroke was scribed into the vessel */
        line[reads++] = (char)ch;

        /* And when the vessel was filled, it was doubled in the forge */
        if (reads + 1 == size) {
            size *= 2;
            char *tmp = realloc(line, size);
            if (!tmp) {
                /* The forge cooled; memory could not be expanded */
                free(line);
                return NULL;
            }
            line = tmp;
        }
    }

    /* If the heavens fell silent before a single jot, the vessel was cast away */
    if (ch == EOF && reads == 0) {
        free(line);
        return NULL;
    }

    /* And the vessel was sealed with the holy terminator */
    line[reads] = '\0';
    return line;
}


//float ** new2d(const int rows, const int cols) {
//
//    float **row = (float**)malloc((rows)*sizeof(float *));
//
//    for(int r = 0; r<rows; r++) {
//        rows[&r] = (float *)malloc((cols)*sizeof(float));
//    }
//    return row;
//}

/* And thus the Lord commanded the crafting of two sacred blocks (optimized) */
float **new2d(const int rows, const int cols) {
    /* Abort on invalid dimensions */
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr,
                "[ERROR] new2d: dimensions denied (rows=%d, cols=%d)\n",
                rows, cols);
        return NULL;
    }

    /* Compute sizes: (rows+1) pointers + rows*cols floats */
    size_t count   = (size_t)rows + 1;
    size_t ptrs_sz = count * sizeof(float*);
    size_t data_sz = (size_t)rows * cols * sizeof(float);

    /* Single allocation for pointer array + data slab */
    void *block = malloc(ptrs_sz + data_sz);
    if (!block) {
        fprintf(stderr,
                "[ERROR] new2d: allocation failed (%zu bytes)\n",
                ptrs_sz + data_sz);
        return NULL;
    }

    /* Split block into pointer array and contiguous data */
    float **ptrs = (float**)block;
    float  *data = (float*)((char*)block + ptrs_sz);

    /* Sentinel holds start-of-data for freeing */
    ptrs[0] = data;

    /* Populate row pointers */
    for (int r = 0; r < rows; ++r) {
        ptrs[r + 1] = data + (size_t)r * cols;
    }

    /* Return base so ptrs[-1]==data and ptrs[0..rows-1] are the rows */
    return ptrs + 1;
}


/* And thus the legions of Data were called forth into being */
Data ndata(const int nips, const int nops, const int rows) {
    /* And lo, the vessel of Data was forged with null pointers and the appointed ranks */
    Data data = { NULL, NULL, nips, nops, rows };

    /* If any host be counted as zero or less, the rites are forbidden */
    if (nips <= 0 || nops <= 0 || rows <= 0) {
        fprintf(stderr,
                "[ERROR] ndata: invalid dimensions (nips=%d, nops=%d, rows=%d)\n",
                nips, nops, rows);
        return data;
    }

    /* And the front-line cohorts were arrayed in contiguous order */
    data.in = new2d(rows, nips);
    if (!data.in) {
        fprintf(stderr,
                "[ERROR] ndata: failed to allocate input matrix\n");
        return data;
    }

    /* And the reserve cohorts were likewise arrayed in steadfast formation */
    data.tg = new2d(rows, nops);
    if (!data.tg) {
        fprintf(stderr,
                "[ERROR] ndata: failed to allocate target matrix\n");
        dfree(&data);      /* The fallen are gathered back into the void */
        data.in = NULL;   /* The front-line is undone as well */
        return data;
    } 

    /* And thus the armies stood ready upon the field of data */
    return data;
}


/***************************************** TO USE IN PLACE OF strtof() ***************************************** */
/* fast inline float‐parser: handles [+/-]?[0-9]*[.[0-9]*][eE[+/-]?[0-9]+] */
static inline float fast_atof(const char *p, char **endptr) {
    // sign
    bool neg = false;
    if (*p == '-' || *p == '+') {
        neg = (*p == '-');
        p++;
    }

    // integer part
    unsigned int ip = 0;
    while (*p >= '0' && *p <= '9') {
        ip = ip * 10 + (*p++ - '0');
    }
    float val = (float)ip;

    // fraction
    if (*p == '.') {
        p++;
        float base = 0.1f;
        while (*p >= '0' && *p <= '9') {
            val += (*p++ - '0') * base;
            base *= 0.1f;
        }
    }

    // exponent
    if (*p == 'e' || *p == 'E') {
        p++;
        bool exp_neg = false;
        if (*p == '-' || *p == '+') {
            exp_neg = (*p == '-');
            p++;
        }
        int exp = 0;
        while (*p >= '0' && *p <= '9') {
            exp = exp * 10 + (*p++ - '0');
        }
        // compute 10^exp
        float pow10 = 1.0f;
        for (int i = 0; i < exp; i++) pow10 *= 10.0f;
        val = exp_neg ? val / pow10 : val * pow10;
    }

    if (neg) val = -val;
    *endptr = (char*)p;
    return val;
}

/* And thus the verse of parsing was inscribed by the Almighty */
void parse(const Data data, char *line, const int row) {
    const int nips  = data.nips;
    const int total = nips + data.nops;

    float *in_row = data.in[row];
    float *tg_row = data.tg[row];

    char *p = line;
    char *next;

    for (int col = 0; col < total; ++col) {
        // skip to start of next token
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;  // no more tokens

        // parse float
        float val = fast_atof(p, &next);

        // assign to input or target
        if (col < nips) {
            in_row[col] = val;
        } else {
            tg_row[col - nips] = val;
        }

        // advance pointer
        p = next;
    }
}


/* helper – only free the block once, then poison the handle */
static inline void free2d_block(float **arr)
{
    if (!arr) return;               /* already freed / was never allocated */
    free(arr - 1);                  /* sentinel lives at arr[-1] */
}

/* ------------------------------------------------------------------ */
/*  Free a Data slab (inputs + targets).                              */
/*                                                                    */
/*  NOTE: split_dataset() and k-fold helpers create alias “views”.    */
/*  Only the original owner should actually free memory. The guards   */
/*  below make dfree() a safe no-op when called on such views.        */
/* ------------------------------------------------------------------ */
/* Free → *and* poison the Data so double-free becomes impossible */
/* idempotent: safe to call repeatedly & from any error-path */
void dfree(Data *d)
{
    if (!d) return;                 /* NULL guard */

    free2d_block(d->in);
    free2d_block(d->tg);

    /* poison everything so future calls are no-ops */
    d->in   = NULL;
    d->tg   = NULL;
    d->nips = d->nops = d->rows = 0;
}



/* And thus the legions were arrayd for redeployment upon the field of training */
void shuffle(const Data d) {
    /* If no host stands or but one cohort remains, the battle lines hold firm */
    if (!d.in || !d.tg || d.rows < 2) {
        return;
    }

    /* And the captains drew forth the pointers to the frontline and reserve cohorts */
    float **in_ptrs = d.in;
    float **tg_ptrs = d.tg;
    int count       = d.rows;

    /* From the last cohort to the first did they choose a comrade to exchange */
    for (int i = count - 1; i > 0; --i) {
        /* The Lord’s random decree selects an index between zero and i inclusive */
        int j = rand() % (i + 1);

        /* And the reserve cohorts were swapped in lockstep */
        float *temp_t = tg_ptrs[i];
        tg_ptrs[i]    = tg_ptrs[j];
        tg_ptrs[j]    = temp_t;

        /* And the frontline cohorts were likewise exchanged */
        float *temp_i = in_ptrs[i];
        in_ptrs[i]    = in_ptrs[j];
        in_ptrs[j]    = temp_i;
    }
}


/* And thus the breach upon the data vault was commanded at the given path */
Data build(const char *path, const int nips, const int nops) {
    /* Open the fortress gates */
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr,
                "Error: could not open data file '%s': %s\n",
                path, strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Count the fallen lines */
    int rows = lns(file);
    if (rows < 0) {
        fprintf(stderr,
                "Error: failed to count lines in file '%s'\n",
                path);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    /* Muster the Data legions */
    Data data = ndata(nips, nops, rows);
    if (!data.in || !data.tg) {
        fprintf(stderr,
                "Error: failed to allocate data matrices (nips=%d, nops=%d, rows=%d)\n",
                nips, nops, rows);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    /* Prepare a single growing buffer for lines */
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;

    /* Parse each row */
    for (int row = 0; row < rows; ++row) {
        len = getline(&line, &cap, file);
        if (len < 0) {
            fprintf(stderr,
                    "Error: failed to read line %d from file '%s'\n",
                    row, path);
            free(line);
            dfree(&data);
            fclose(file);
            exit(EXIT_FAILURE);
        }
        parse(data, line, row);
    }

    /* Clean up */
    free(line);
    fclose(file);

    /* Return the full conscription */
    return data;
}