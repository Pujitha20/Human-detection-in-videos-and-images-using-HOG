function [features,labels] = hog(image_ori,count1,class,features,labels)
im = rgb2gray(image_ori);
Image = double(im);

mask_x = [-1 0 1];
image_grad_x = imfilter(Image,mask_x);%,'same');

mask_y = mask_x';
image_grad_y = imfilter(Image,mask_y);%,'same');

image_grad_mag = sqrt((image_grad_x).^2 + (image_grad_y).^2);
n= norm(image_grad_mag);
image_grad_ang = (atan2(image_grad_y,image_grad_x))*(180/pi);


G=fspecial('gaussian',[8 8],8); 
image_grad_mag = conv2(image_grad_mag,G,'same'); 

total_angle  = 180;
temp = image_grad_ang;
% temp(image_grad_ang(:)<0) = image_grad_ang(image_grad_ang(:)<0)+180;
image_grad_angle=temp;


[rows,cols] = size(image_grad_ang);
cellsize = 8;
num_cell_row = floor(rows/cellsize);
num_cell_cols = floor(cols/cellsize);
num_bins = 9;
binsize = total_angle/num_bins;
ori_bin = zeros(num_cell_row,num_cell_cols,num_bins);

for cell_row=1:num_cell_row
      for cell_col=1:num_cell_cols
              
        for bin=1:num_bins
            cell_row_srt=(cell_row-1)*cellsize+1;
            cell_row_end=(cell_row)*cellsize;
            cell_col_srt=(cell_col-1)*cellsize+1;
            cell_col_end=(cell_col)*cellsize;
            
            temp=zeros(cell_row_end-cell_row_srt+1,cell_col_end-cell_col_srt+1);
           
            for i=cell_row_srt:cell_row_end
                for j=cell_col_srt:cell_col_end
                    if((image_grad_angle(i,j)>=(bin-1)*binsize+1)&&(image_grad_angle(i,j)<(bin)*binsize))
                        temp(i-cell_row_srt+1,j-cell_col_srt+1)=1;
                    end  
                     if(bin>1)
                        if((image_grad_angle(i,j)>=(bin-2)*binsize+1+binsize/2)&&(image_grad_angle(i,j)<(bin-1)*binsize))
                            A(i-cell_row_srt+1,j-cell_col_srt+1)=1-abs(image_grad_angle(i,j)-(bin*binsize-binsize/2))/binsize;
                        end
                    end
                    if(bin<total_angle/binsize)
                        if((image_grad_angle(i,j)>=(bin)*binsize+1)&&(image_grad_angle(i,j)<(bin+1)*binsize-binsize/2))
                            A(i-cell_row_srt+1,j-cell_col_srt+1)=1-abs(image_grad_angle(i,j)-(bin*binsize-binsize/2))/binsize;
                        end
                    end                  
                
                end
            end
                
            ori_bin(cell_row,cell_col ,bin) = sum(sum(temp.* image_grad_mag(cell_row_srt:cell_row_end,cell_col_srt:cell_col_end)));
        end
                   
%             subplot(num_cell_row,num_cell_cols, (cell_row-1)*num_cell_cols+cell_col);
%             vector=permute(ori_bin(cell_row,cell_col,:),[3,2,1]);
%             bar(vector);
%             set(gca,'xtick',[],'ytick',[]);
      end
end

stride=4;
block_row=num_cell_row-1;
block_col=num_cell_cols-1;

ori_bin_blocks=zeros(block_row,block_col,stride*total_angle/binsize);
for bl_row=1:block_row
    for bl_col=1:block_col
       
        block_vector=zeros(1,stride*total_angle/binsize);
        for i=1:2
            for j=1:2
                cellI=(bl_row-1)+i;
                cellJ=(bl_col-1)+j;
                vector=permute(ori_bin(cellI,cellJ,:),[3,2,1]);
                cellblocks=((i-1)*2+j);
                block_vector((cellblocks-1)*(total_angle/binsize)+1:(cellblocks)*(total_angle/binsize))=vector;
                
            end
        end
        norm_block_vec=block_vector ./norm(block_vector);
        norm_block_vec(norm_block_vec(:)>0.2)=0.2;
        norm_block_vec=norm_block_vec ./ norm(norm_block_vec);
        
        ori_bin_blocks(bl_row,bl_col,:)=norm_block_vec;
       
    end
end
h=size(ori_bin_blocks);

 blocks=zeros(1,size(ori_bin_blocks,1)*size(ori_bin_blocks,2)*size(ori_bin_blocks,3));
    for i=1:size(ori_bin_blocks,1)
        for j=1:size(ori_bin_blocks,2)
            for k=1:size(ori_bin_blocks,3)
                blocks((i-1)*size(ori_bin_blocks,2)*size(ori_bin_blocks,3)+(j-1)*size(ori_bin_blocks,3)+k)=ori_bin_blocks(i,j,k);
            end
        end
    end
features(count1, :) = blocks;
labels(count1) = class; 

