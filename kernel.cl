


__kernel void btm(__global  const uchar *image, __global unsigned char *result,  int rows,  int cols,int maxBtm,int margin,int minBtm){

            int i = get_global_id(0);
            int j = get_global_id(1);
            int btm;
            int auxk,auxl;
            int br = 0;

                if((i > 0 && i < rows-1) && (j > 0 && j < cols-1))
                {

                    for(int k= i-1; k<= i+1; k++)
                    {
                        for(int l=j-1;l<= j+1; l++)
                        {

                                    btm=0;
                                    auxk = k;
                                    auxl = l;
                                    
                                    while(btm <= maxBtm)
                                    {
                                        if(image[i*cols+j]-image[k * cols+ l] <= -margin || image[i*cols+j]-image[k * cols + l] >= margin)
                                        {
                                            
                                            btm++;
                                            k = k+(k-i);
                                            l = l+(l-j);
                                           if((k <=0 || k>=rows-1 || l<=0 || l >= cols-1)){
                                                    break;
                                            }    
                                            
                                        }
                                        else
                                        {
                                            break;
                                        }
                                    }

                                    k = auxk;
                                    l = auxl;
                                   

                                    if(btm > minBtm && btm <= maxBtm)
                                    {
                                        result[i*cols+j] = 255;
                                        br = 1;
                                        break;
                                    }
                        }
                        if( br == 1)
                        {
                                break;
                        }
                        
                    }
                }     
}
