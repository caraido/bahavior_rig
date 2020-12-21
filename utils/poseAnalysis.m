function resultStruct = poseAnalysis(dlc_filename, likelihood_thres, outer_r_pixel, inner_r_pixel, arena_center)
gaze_angle = 32.8; %degrees
% arena_center = [562, 437]; %pixels
if isa(arena_center,'cell')
    arena_center = cell2mat(arena_center);
    arena_center=cast(arena_center,'double');
end

D = dlmread(dlc_filename, ',', 3, 1);
Nframes = size(D,1);

snout_x = D(:,1);
snout_y = D(:,2);
snout_L = D(:,3);

earL_x = D(:,4);
earL_y = D(:,5);
earL_L = D(:,6);

earR_x = D(:,7);
earR_y = D(:,8);
earR_L = D(:,9);

tailbase_x = D(:,10);
tailbase_y = D(:,11);
tailbase_L = D(:,12);

ok_ind = snout_L > likelihood_thres & ...
    earL_L > likelihood_thres & ...
    earR_L > likelihood_thres & ...
    tailbase_L > likelihood_thres;

Nframes_passed = sum(ok_ind);

msg = sprintf('%d of %d frames passed likelihood threshold', Nframes_passed, Nframes);
disp(msg);

snout_x = snout_x(ok_ind);
snout_y = snout_y(ok_ind);
earL_x = earL_x(ok_ind);
earL_y = earL_y(ok_ind);
earR_x = earR_x(ok_ind);
earR_y = earR_y(ok_ind);
tailbase_x = tailbase_x(ok_ind);
tailbase_y = tailbase_y(ok_ind);

headbase_x = mean([earL_x, earR_x],2);
headbase_y = mean([earL_y, earR_y],2);

resultStruct.headbase_x = headbase_x;
resultStruct.headbase_y = headbase_y;

dx = snout_x-headbase_x;
dy = snout_y-headbase_y;
gazeFront_lines = [headbase_x, headbase_y, dx, dy];


theta_rad = cart2pol(dx,dy);

new_theta_L = deg2rad(rad2deg(theta_rad) + gaze_angle);
new_theta_R = deg2rad(rad2deg(theta_rad) - gaze_angle);

[dx_L, dy_L] = pol2cart(new_theta_L, 1);
[dx_R, dy_R] = pol2cart(new_theta_R, 1);

gazeLeft_lines = [headbase_x, headbase_y, dx_L, dy_L];
gazeRight_lines = [headbase_x, headbase_y, dx_R, dy_R];

%center_to_snout = sqrt((snout_x - arena_center(1)).^2 + (snout_y - arena_center(2)).^2);

circ_inner = double([arena_center, inner_r_pixel]);
circ_outer = double([arena_center, outer_r_pixel]);

intersect_outer_left = intersectLineCircle(gazeLeft_lines, repmat(circ_outer, [Nframes_passed, 1]));
%intersect_inner_left = intersectLineCircle(gazeLeft_lines, repmat(circ_inner, [Nframes_passed, 1]));

intersect_outer_right = intersectLineCircle(gazeRight_lines, repmat(circ_outer, [Nframes_passed, 1]));
%intersect_inner_right = intersectLineCircle(gazeRight_lines, repmat(circ_inner, [Nframes_passed, 1]));

outer_angles_left = zeros(Nframes_passed,1);
inner_angles_left = zeros(Nframes_passed,1);

outer_angles_right = zeros(Nframes_passed,1);
inner_angles_right = zeros(Nframes_passed,1);

%left eye
for i=1:Nframes_passed
   I_outer = intersect_outer_left(:,:,i);
   %I_inner = intersect_inner_left(:,:,i);
   
   %figure out correct outer intersection
   if isPointOnRay(I_outer(1,:), gazeLeft_lines(i,:))
      outer_pt = I_outer(1,:);
   else
      outer_pt = I_outer(2,:);
   end
   
   %figure out closest inner intersection
   I_inner = intersectLineCircle(gazeLeft_lines(i,:), circ_inner);
   if sum(isnan(I_inner),'all')
       %inner is not on the ray, so outer is correct
       dx = outer_pt(1) - arena_center(1);
       dy = outer_pt(2) - arena_center(2);
       outer_angles_left(i) = rad2deg(cart2pol(dx,dy));      
       inner_angles_left(i) = nan;
   elseif ~isPointOnRay(I_inner(1,:), gazeLeft_lines(i,:)) %inner intersection on wrong side
       %so outer is correct
       dx = outer_pt(1) - arena_center(1);
       dy = outer_pt(2) - arena_center(2);
       outer_angles_left(i) = rad2deg(cart2pol(dx,dy));    
       inner_angles_left(i) = nan;
   else %outer intersection occludded by inner
       L1 = edgeLength(gazeLeft_lines(i,1:2), I_inner(1,:));
       L2 = edgeLength(gazeLeft_lines(i,1:2), I_inner(2,:));
       if L1<L2
           dx = I_inner(1,1) - arena_center(1);
           dy = I_inner(1,2) - arena_center(2);
       else
           dx = I_inner(2,1) - arena_center(1); 
           dy = I_inner(2,2) - arena_center(2);
       end
       inner_angles_left(i) = rad2deg(cart2pol(dx,dy));  
       outer_angles_left(i) = nan;
   end
       
end

%right eye
for i=1:Nframes_passed
   I_outer = intersect_outer_right(:,:,i);
   
   %figure out correct outer intersection
   if isPointOnRay(I_outer(1,:), gazeRight_lines(i,:))
      outer_pt = I_outer(1,:);
   else
      outer_pt = I_outer(2,:);
   end
   
   %figure out closest inner intersection
   I_inner = intersectLineCircle(gazeRight_lines(i,:), circ_inner);
   if sum(isnan(I_inner),'all')
       %inner is not on the ray, so outer is correct
       dx = outer_pt(1) - arena_center(1); 
       dy = outer_pt(2) - arena_center(2);
       outer_angles_right(i) = rad2deg(cart2pol(dx,dy));      
       inner_angles_right(i) = nan;
   elseif ~isPointOnRay(I_inner(1,:), gazeRight_lines(i,:)) %inner intersection on wrong side
       %so outer is correct
       dx = outer_pt(1) - arena_center(1); 
       dy = outer_pt(2) - arena_center(2);
       outer_angles_right(i) = rad2deg(cart2pol(dx,dy));    
       inner_angles_right(i) = nan;
   else %outer intersection occludded by inner
       L1 = edgeLength(gazeRight_lines(i,1:2), I_inner(1,:));
       L2 = edgeLength(gazeRight_lines(i,1:2), I_inner(2,:));
       if L1<L2
           dx = I_inner(1,1) - arena_center(1); 
           dy = I_inner(1,2) - arena_center(2);
       else
           dx = I_inner(2,1) - arena_center(1); 
           dy = I_inner(2,2) - arena_center(2);
       end
       inner_angles_right(i) = rad2deg(cart2pol(dx,dy));  
       outer_angles_right(i) = nan;
   end
       
end

resultStruct.outer_angles_left = 180+outer_angles_left;
resultStruct.inner_angles_left = 180+inner_angles_left;
resultStruct.outer_angles_right = 180+outer_angles_right;
resultStruct.inner_angles_right = 180+inner_angles_right;




