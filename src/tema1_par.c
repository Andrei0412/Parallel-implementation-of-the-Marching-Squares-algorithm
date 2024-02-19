/*
* APD Tema 1
* Octombrie 2023
*/

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

// structure for thread's parameters.
typedef struct {
    int thread_id;
    int nr_threads;
    unsigned char **grid;
    pthread_barrier_t *barrier;
    ppm_image **image; 
    ppm_image **new_image;
    ppm_image **contour_map;
} thread_data;

ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

unsigned char **init_grid(ppm_image *image) {
    int p, q;

    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        p = image->x / STEP;
        q = image->y / STEP;
    } else {
        p = RESCALE_X / STEP;
        q = RESCALE_Y / STEP;
    }

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    return grid;
}

void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

void *thread_function(void* arg) {
    thread_data* thread_info = (thread_data*)arg;
    int start, end, p, q;

    /*
    * Check if the image needs to be rescaled, if so then allocate memory for a new one with 1 thread.
    * Computations are made by all threads.
    */

    if ((*thread_info->image)->x > RESCALE_X && (*thread_info->image)->y > RESCALE_Y) {
        if (thread_info->thread_id == 0) {
            *(thread_info->new_image) = (ppm_image *)malloc(sizeof(ppm_image));
            if (!(*thread_info->new_image)) {
                fprintf(stderr, "Unable to allocate memory\n");
                exit(1);
            }   

            (*thread_info->new_image)->x = RESCALE_X;
            (*thread_info->new_image)->y = RESCALE_Y;

            (*thread_info->new_image)->data = (ppm_pixel*)malloc(RESCALE_X * RESCALE_Y * sizeof(ppm_pixel));
            if (!(*thread_info->new_image)->data) {
                fprintf(stderr, "Unable to allocate memory\n");
                exit(1);
            }
        }

        // Wait for the memory to be allocated before starting the computations.
        pthread_barrier_wait(thread_info->barrier);

        start = thread_info->thread_id * (double)(*thread_info->new_image)->y / thread_info->nr_threads;
        end = fmin((thread_info->thread_id + 1) * (double)(*thread_info->new_image)->y/ thread_info->nr_threads, (*thread_info->new_image)->y);
        uint8_t sample[3];

        for (int i = 0; i < (*thread_info->new_image)->x; i++) {
            for (int j = start; j < end; j++) {
                float u = (float)i / (float)((*thread_info->new_image)->x - 1);
                float v = (float)j / (float)((*thread_info->new_image)->y - 1);
                sample_bicubic((*thread_info->image), u, v, sample);

                (*thread_info->new_image)->data[i * (*thread_info->new_image)->y + j].red = sample[0];
                (*thread_info->new_image)->data[i * (*thread_info->new_image)->y + j].green = sample[1];
                (*thread_info->new_image)->data[i * (*thread_info->new_image)->y + j].blue = sample[2];
            }
        }
        pthread_barrier_wait(thread_info->barrier);

        /*
        * Make the original image pointer indicate towards the new allocated memory
        * so that the code below works for either original or rescaled image and
        * use an auxiliary pointer to free the original memory if we needed to rescale.
        * We only need 1 thread to perform this.
        */

        if (thread_info->thread_id == 0) {
            ppm_image *aux;
            aux = (*thread_info->image);
            aux->data = (*thread_info->image)->data;

            (*thread_info->image) = (*thread_info->new_image);
            (*thread_info->image)->data = (*thread_info->new_image)->data;

            free(aux->data);
            free(aux); 
        }
        pthread_barrier_wait(thread_info->barrier);
    }

    /*
    * From here onwards we follow the 2 steps of the algorithm.
    * Compute the portion for each thread before entering the loop
    * Use a barrier after each loop is done to make sure that the data is ready.
    * Proceed to the next step.
    */

    p = (*thread_info->image)->x / STEP;
    q = (*thread_info->image)->y / STEP;

    start = thread_info->thread_id * (double)p / thread_info->nr_threads;
    end = fmin((thread_info->thread_id + 1) * (double)p / thread_info->nr_threads, p);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = (*thread_info->image)->data[i * STEP * (*thread_info->image)->y + j * STEP];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > SIGMA) {
                thread_info->grid[i][j] = 0;
            } else {
                thread_info->grid[i][j] = 1;
            }
        }
    }
    pthread_barrier_wait(thread_info->barrier);

    if (thread_info->thread_id == 0)
        thread_info->grid[p][q] = 0;
    pthread_barrier_wait(thread_info->barrier);

    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = (*thread_info->image)->data[i * STEP * (*thread_info->image)->y + (*thread_info->image)->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            thread_info->grid[i][q] = 0;
        } else {
            thread_info->grid[i][q] = 1;
        }
    }
    pthread_barrier_wait(thread_info->barrier);

    start = thread_info->thread_id * (double)q / thread_info->nr_threads;
    end = fmin((thread_info->thread_id + 1) * (double)q / thread_info->nr_threads, q);

    for (int j = start; j < end; j++) {
        ppm_pixel curr_pixel = (*thread_info->image)->data[((*thread_info->image)->x - 1) * (*thread_info->image)->y + j * STEP];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            thread_info->grid[p][j] = 0;
        } else {
            thread_info->grid[p][j] = 1;
        }
    }
    pthread_barrier_wait(thread_info->barrier);

    start = thread_info->thread_id * (double)p / thread_info->nr_threads;
    end = fmin((thread_info->thread_id + 1) * (double)p / thread_info->nr_threads, p);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * thread_info->grid[i][j] + 4 * thread_info->grid[i][j + 1] + 
                            2 * thread_info->grid[i + 1][j + 1] + 1 * thread_info->grid[i + 1][j];
            update_image((*thread_info->image), thread_info->contour_map[k], i * STEP, j * STEP);
        }
    }
    pthread_barrier_wait(thread_info->barrier);

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    // Read input image and nr of threads and initialize a new image.
    ppm_image *image = read_ppm(argv[1]);
    ppm_image *new_image = NULL;
    int nr_threads = atoi(argv[3]);

    // Initialize information regarding the threads.
    pthread_t threads[nr_threads];
    thread_data thread_info[nr_threads];
    pthread_barrier_t barrier;
    int r;

    // Initialize the barrier.
    if (pthread_barrier_init(&barrier, NULL, nr_threads) != 0)
	{
		printf("Error can't initalize barrier");
		return 1;
	}

    // Initialize contour map and the grid
    ppm_image **contour_map = init_contour_map();
    unsigned char **grid = init_grid(image);

    // Start the threads and give them the arguments.
    for (int id = 0; id < nr_threads; id++) {
        thread_info[id].thread_id = id;
        thread_info[id].nr_threads = nr_threads;
        thread_info[id].grid = grid;
        thread_info[id].barrier = &barrier;
        thread_info[id].image = &image;
        thread_info[id].new_image = &new_image;
        thread_info[id].contour_map = contour_map;

        r = pthread_create(&threads[id], NULL, thread_function,  (void *) &thread_info[id]);

        if (r) {
            fprintf(stderr, "Error creating thread %d\n", id);
            exit(1);
        }
    }

    //Join the threads.
    for (int id = 0; id < nr_threads; id++) {
        r = pthread_join(threads[id], NULL);
 
        if (r) {
            printf("Eroare waiting for thread %d\n", id);
            exit(-1);
        }
    }

    // Write output
    write_ppm(image, argv[2]);

    // Free resources and destroy the barrier.
    free_resources(image, contour_map, grid, STEP);
    pthread_barrier_destroy(&barrier);

    return 0;
}
